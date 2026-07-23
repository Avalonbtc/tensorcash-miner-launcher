#!/usr/bin/env bash
# Shared TensorCash runtime-profile policy.  This file is sourced by both
# launchers so Docker and native mode cannot disagree about a card's model
# precision or minimum usable VRAM.

tensorcash_bf16_single_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP1_MIN_MIB:-22000}}"
}

tensorcash_fp8_single_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_FP8_MIN_VRAM_MIB:-11500}"
}

tensorcash_fp8_tp2_min_vram_mib() {
  # FP8 shards the 8B model across two ranks. 6 GiB cards expose 6144 MiB,
  # leaving vLLM's runtime capacity probe as the final admission authority.
  printf '%s\n' "${TENSORCASH_FP8_TP2_MIN_VRAM_MIB:-6000}"
}

tensorcash_bf16_tp2_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_TP2_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP2_MIN_MIB:-11000}}"
}

tensorcash_bf16_tp4_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_TP4_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP4_MIN_MIB:-7500}}"
}

# A 12 GiB FP8 card can hold the roughly 8 GiB quantized model, but it has
# much less room for initial KV-cache/profile allocations than a 16 GiB card.
# These values reduce only that low-VRAM TP=1 bootstrap; 16 GiB and larger
# cards retain the normal 2048-token, 0.89 memory-utilization profile.
tensorcash_low_vram_fp8_tp1() {
  local memory="$1" tensor_parallel_size="$2" precision="$3"
  local ceiling="${TENSORCASH_LOW_VRAM_FP8_MAX_VRAM_MIB:-15000}"
  [[ "$memory" =~ ^[1-9][0-9]*$ && "$tensor_parallel_size" == 1 && "$precision" == fp8 && "$ceiling" =~ ^[1-9][0-9]*$ ]] || return 1
  (( memory < ceiling ))
}

tensorcash_low_vram_fp8_max_model_len() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_MAX_MODEL_LEN:-512}"
}

tensorcash_low_vram_fp8_gpu_mem_util() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_GPU_MEM_UTIL:-0.78}"
}

tensorcash_low_vram_fp8_concurrency_cap() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_CONCURRENCY_CAP:-64}"
}

tensorcash_low_vram_fp8_concurrency_start() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_CONCURRENCY_START:-4}"
}

tensorcash_low_vram_fp8_concurrency_step() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_CONCURRENCY_STEP:-4}"
}

tensorcash_low_vram_fp8_max_batched_tokens() {
  printf '%s\n' "${TENSORCASH_LOW_VRAM_FP8_MAX_BATCHED_TOKENS:-1024}"
}

tensorcash_precision_mode() {
  local mode="${TENSORCASH_MODEL_PRECISION:-auto}"
  local legacy_quantization="${TENSORCASH_VLLM_QUANTIZATION:-}"

  case "$mode" in
    auto|bf16|fp8) ;;
    *)
      echo "TENSORCASH_MODEL_PRECISION must be auto, bf16, or fp8." >&2
      return 1
      ;;
  esac

  # Keep the older direct vLLM option working as an explicit FP8 override, but
  # do not permit arbitrary quantizers for the chain-pinned mining profile.
  case "$legacy_quantization" in
    ''|fp8) ;;
    *)
      echo "TENSORCASH_VLLM_QUANTIZATION supports only fp8; use TENSORCASH_MODEL_PRECISION." >&2
      return 1
      ;;
  esac
  if [[ "$legacy_quantization" == fp8 ]]; then
    [[ "$mode" != bf16 ]] || {
      echo "TENSORCASH_MODEL_PRECISION=bf16 conflicts with TENSORCASH_VLLM_QUANTIZATION=fp8." >&2
      return 1
    }
    mode=fp8
  fi
  printf '%s\n' "$mode"
}

tensorcash_validate_vram_thresholds() {
  local bf16 fp8 fp8_tp2 tp2 tp4
  bf16="$(tensorcash_bf16_single_min_vram_mib)"
  fp8="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2="$(tensorcash_fp8_tp2_min_vram_mib)"
  tp2="$(tensorcash_bf16_tp2_min_vram_mib)"
  tp4="$(tensorcash_bf16_tp4_min_vram_mib)"
  [[ "$bf16" =~ ^[1-9][0-9]*$ && "$fp8" =~ ^[1-9][0-9]*$ && \
     "$fp8_tp2" =~ ^[1-9][0-9]*$ && "$tp2" =~ ^[1-9][0-9]*$ && \
     "$tp4" =~ ^[1-9][0-9]*$ ]] || {
    echo "TensorCash VRAM thresholds must be positive MiB integers." >&2
    return 1
  }
  (( bf16 > fp8 && fp8 >= tp2 && tp2 > tp4 && fp8 > fp8_tp2 )) || {
    echo "TensorCash VRAM thresholds must satisfy BF16 TP1 > FP8 TP1 >= BF16 TP2 > BF16 TP4 and FP8 TP1 > FP8 TP2." >&2
    return 1
  }
}

tensorcash_min_single_vram_mib() {
  local mode
  mode="$(tensorcash_precision_mode)" || return 1
  case "$mode" in
    bf16) tensorcash_bf16_single_min_vram_mib ;;
    auto|fp8) tensorcash_fp8_single_min_vram_mib ;;
  esac
}

# Emit the effective model profile for a TP group. In auto mode, 12/16 GiB
# cards become independent FP8 miners; legacy sub-11.5 GiB TP groups retain
# BF16, and 24 GiB-plus single cards stay on canonical BF16.
tensorcash_resolve_precision() {
  local memory="$1" tensor_parallel_size="$2" mode bf16 fp8 fp8_tp2 tp2 tp4
  [[ "$memory" =~ ^[1-9][0-9]*$ ]] || {
    echo "TensorCash GPU VRAM must be a positive MiB integer." >&2
    return 1
  }
  [[ "$tensor_parallel_size" =~ ^(1|2|4|8)$ ]] || {
    echo "TensorCash supports TP=1, 2, 4, or 8 only." >&2
    return 1
  }

  mode="$(tensorcash_precision_mode)" || return 1
  tensorcash_validate_vram_thresholds || return 1
  bf16="$(tensorcash_bf16_single_min_vram_mib)"
  fp8="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2="$(tensorcash_fp8_tp2_min_vram_mib)"
  tp2="$(tensorcash_bf16_tp2_min_vram_mib)"
  tp4="$(tensorcash_bf16_tp4_min_vram_mib)"

  case "$mode" in
    auto)
      case "$tensor_parallel_size" in
        1)
          if (( memory >= bf16 )); then
            printf 'bf16\n'
          elif (( memory >= fp8 )); then
            printf 'fp8\n'
          else
            echo "TP=1 needs >=${fp8} MiB for FP8 or >=${bf16} MiB for BF16; GPU has ${memory} MiB." >&2
            return 1
          fi
          ;;
        2)
          if (( memory >= tp2 )); then
            printf 'bf16\n'
          elif (( memory >= fp8_tp2 )); then
            printf 'fp8\n'
          else
            echo "TP=2 needs >=${fp8_tp2} MiB per GPU for FP8 or >=${tp2} MiB for BF16; group minimum is ${memory} MiB." >&2
            return 1
          fi
          ;;
        4|8)
          (( memory >= tp4 )) || {
            echo "TP=${tensor_parallel_size} BF16 needs >=${tp4} MiB per GPU; group minimum is ${memory} MiB." >&2
            return 1
          }
          printf 'bf16\n'
          ;;
      esac
      ;;
    bf16)
      case "$tensor_parallel_size" in
        1) (( memory >= bf16 )) ;;
        2) (( memory >= tp2 )) ;;
        4|8) (( memory >= tp4 )) ;;
      esac || {
        echo "Forced BF16 does not fit TP=${tensor_parallel_size} with ${memory} MiB per GPU." >&2
        return 1
      }
      printf 'bf16\n'
      ;;
    fp8)
      case "$tensor_parallel_size" in
        1) (( memory >= fp8 )) ;;
        2) (( memory >= fp8_tp2 )) ;;
        4|8)
          echo "Forced FP8 supports TP=1 or TP=2 only; use BF16 for TP=${tensor_parallel_size}." >&2
          return 1
          ;;
      esac || {
        echo "Forced FP8 does not fit TP=${tensor_parallel_size} with ${memory} MiB per GPU." >&2
        return 1
      }
      printf 'fp8\n'
      ;;
  esac
}

tensorcash_vllm_quantization() {
  case "$1" in
    bf16) printf '\n' ;;
    fp8) printf 'fp8\n' ;;
    *)
      echo "Unknown TensorCash precision profile: $1" >&2
      return 1
      ;;
  esac
}

tensorcash_validate_gpu_mem_util() {
  local precision="$1" value="$2" minimum
  case "$precision" in
    auto|fp8) minimum=0.40 ;;
    bf16) minimum=0.50 ;;
    *)
      echo "Unknown TensorCash precision profile: $precision" >&2
      return 1
      ;;
  esac
  [[ "$value" =~ ^([0-9]+)(\.[0-9]+)?$ ]] || {
    echo "GPU_MEM_UTIL must be a decimal number." >&2
    return 1
  }
  awk -v value="$value" -v minimum="$minimum" \
    'BEGIN { exit !(value >= minimum && value <= 0.95) }' || {
      echo "GPU_MEM_UTIL must be between ${minimum} and 0.95 for ${precision}." >&2
      return 1
    }
}
