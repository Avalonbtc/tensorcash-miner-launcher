#!/usr/bin/env bash
# Shared TensorCash runtime-profile policy.  This file is sourced by both
# launchers so Docker and native mode cannot disagree about a card's model
# precision or minimum usable VRAM.

tensorcash_bf16_single_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP1_MIN_MIB:-22000}}"
}

tensorcash_fp8_single_min_vram_mib() {
  # The public image converts the BF16 safetensors checkpoint to FP8 while
  # constructing the model. That load-time peak exceeds 12 GiB before vLLM
  # reserves any KV cache, so TP=1 FP8 needs a real 15 GiB minimum.
  printf '%s\n' "${TENSORCASH_FP8_MIN_VRAM_MIB:-15000}"
}

tensorcash_fp8_tp2_min_vram_mib() {
  # FP8 shards the 8B model across two ranks. 6 GiB cards expose 6144 MiB,
  # leaving vLLM's runtime capacity probe as the final admission authority.
  printf '%s\n' "${TENSORCASH_FP8_TP2_MIN_VRAM_MIB:-6000}"
}

tensorcash_static_fp8_tp1_min_vram_mib() {
  # A serialized FP8 snapshot skips the BF16-to-FP8 construction peak. Keep a
  # small, explicit 12 GiB floor for remaining BF16 tensors and workspaces.
  printf '%s\n' "${TENSORCASH_STATIC_FP8_TP1_MIN_VRAM_MIB:-12000}"
}

tensorcash_static_fp8_tp1_available() {
  [[ "${TENSORCASH_STATIC_FP8_TP1_AVAILABLE:-false}" == true ]]
}

tensorcash_can_use_static_fp8_tp1() {
  local memory="$1" static_min
  [[ "$memory" =~ ^[1-9][0-9]*$ ]] || return 1
  tensorcash_static_fp8_tp1_available || return 1
  static_min="$(tensorcash_static_fp8_tp1_min_vram_mib)"
  # Once a TP=1 group has resolved to FP8, always prefer the serialized
  # checkpoint from the 12 GiB floor upward. Loading BF16 weights and
  # quantizing them online can still exceed a 16 GiB card before KV cache
  # allocation, whereas the immutable FP8 artifact never has that peak.
  (( memory >= static_min ))
}

# Whether this host should fetch the serialized checkpoint before planning TP=1
# groups. Auto mode needs it below the BF16 tier; an explicit FP8 selection
# needs it at every supported single-GPU tier. Availability is intentionally
# not consulted here because callers use this to decide whether to download it.
tensorcash_static_fp8_tp1_download_needed() {
  local memory="$1" static_min bf16_min mode
  [[ "$memory" =~ ^[1-9][0-9]*$ ]] || return 1
  mode="$(tensorcash_precision_mode)" || return 1
  [[ "$mode" != bf16 ]] || return 1
  static_min="$(tensorcash_static_fp8_tp1_min_vram_mib)"
  bf16_min="$(tensorcash_bf16_single_min_vram_mib)"
  (( memory >= static_min )) || return 1
  [[ "$mode" == fp8 ]] || (( memory < bf16_min ))
}

tensorcash_bf16_tp2_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_TP2_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP2_MIN_MIB:-11000}}"
}

tensorcash_bf16_tp4_min_vram_mib() {
  printf '%s\n' "${TENSORCASH_BF16_TP4_MIN_VRAM_MIB:-${TENSORCASH_AUTO_TP4_MIN_MIB:-7500}}"
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
  local bf16 fp8 fp8_tp2 static_fp8 tp2 tp4
  bf16="$(tensorcash_bf16_single_min_vram_mib)"
  fp8="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2="$(tensorcash_fp8_tp2_min_vram_mib)"
  static_fp8="$(tensorcash_static_fp8_tp1_min_vram_mib)"
  tp2="$(tensorcash_bf16_tp2_min_vram_mib)"
  tp4="$(tensorcash_bf16_tp4_min_vram_mib)"
  [[ "$bf16" =~ ^[1-9][0-9]*$ && "$fp8" =~ ^[1-9][0-9]*$ && \
     "$fp8_tp2" =~ ^[1-9][0-9]*$ && "$static_fp8" =~ ^[1-9][0-9]*$ && "$tp2" =~ ^[1-9][0-9]*$ && \
     "$tp4" =~ ^[1-9][0-9]*$ ]] || {
    echo "TensorCash VRAM thresholds must be positive MiB integers." >&2
    return 1
  }
  (( bf16 > fp8 && fp8 > static_fp8 && static_fp8 >= tp2 && tp2 > tp4 && fp8 > fp8_tp2 )) || {
    echo "TensorCash VRAM thresholds must satisfy BF16 TP1 > FP8 TP1 > static FP8 TP1 >= BF16 TP2 > BF16 TP4 and FP8 TP1 > FP8 TP2." >&2
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

# Emit the effective model profile for a TP group. In auto mode, 6/8 GiB pairs
# resolve to ordinary FP8 TP=2, 12--21.9 GiB single cards resolve to FP8 and
# use the validated serialized snapshot when available, and >=22 GiB single
# cards stay on canonical BF16.
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
          elif (( memory >= fp8 )) || tensorcash_can_use_static_fp8_tp1 "$memory"; then
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
        1) (( memory >= fp8 )) || tensorcash_can_use_static_fp8_tp1 "$memory" ;;
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
