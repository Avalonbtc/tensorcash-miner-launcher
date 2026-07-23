#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../runtime-profile.sh
source "$script_dir/../runtime-profile.sh"

assert_eq() {
  local expected="$1" actual="$2" label="$3"
  [[ "$expected" == "$actual" ]] || {
    echo "FAIL: $label: expected '$expected', got '$actual'" >&2
    exit 1
  }
}

unset TENSORCASH_MODEL_PRECISION TENSORCASH_VLLM_QUANTIZATION
assert_eq fp8 "$(tensorcash_resolve_precision 12282 1)" '12 GiB auto profile'
assert_eq fp8 "$(tensorcash_resolve_precision 16384 1)" '16 GiB auto profile'
assert_eq bf16 "$(tensorcash_resolve_precision 24564 1)" '24 GiB auto profile'
tensorcash_low_vram_fp8_tp1 12282 1 fp8 || {
  echo 'FAIL: 12 GiB FP8 TP1 must receive the low-VRAM startup guard' >&2
  exit 1
}
if tensorcash_low_vram_fp8_tp1 16384 1 fp8 || tensorcash_low_vram_fp8_tp1 12282 2 fp8; then
  echo 'FAIL: the low-VRAM FP8 guard must apply only to 12 GiB-class TP1 groups' >&2
  exit 1
fi
assert_eq 512 "$(tensorcash_low_vram_fp8_max_model_len)" 'low-VRAM FP8 context'
assert_eq 0.78 "$(tensorcash_low_vram_fp8_gpu_mem_util)" 'low-VRAM FP8 memory utilization'
assert_eq 64 "$(tensorcash_low_vram_fp8_concurrency_cap)" 'low-VRAM FP8 concurrency cap'
assert_eq fp8 "$(tensorcash_resolve_precision 6144 2)" '6 GiB FP8 TP2 profile'
assert_eq fp8 "$(tensorcash_resolve_precision 8192 2)" '8 GiB FP8 TP2 profile'
assert_eq bf16 "$(tensorcash_resolve_precision 11000 2)" 'legacy TP2 BF16 profile'
assert_eq bf16 "$(tensorcash_resolve_precision 8192 4)" 'legacy TP4 BF16 profile'

TENSORCASH_MODEL_PRECISION=fp8
assert_eq fp8 "$(tensorcash_resolve_precision 12282 1)" 'forced FP8 profile'
TENSORCASH_MODEL_PRECISION=bf16
assert_eq bf16 "$(tensorcash_resolve_precision 12282 2)" 'forced TP2 BF16 profile'
if tensorcash_resolve_precision 12282 1 >/dev/null 2>&1; then
  echo 'FAIL: forced BF16 must reject a 12 GiB TP1 card' >&2
  exit 1
fi

unset TENSORCASH_MODEL_PRECISION
TENSORCASH_VLLM_QUANTIZATION=fp8
assert_eq fp8 "$(tensorcash_resolve_precision 24564 1)" 'legacy FP8 override'
unset TENSORCASH_VLLM_QUANTIZATION

echo 'runtime profile tests: OK'
