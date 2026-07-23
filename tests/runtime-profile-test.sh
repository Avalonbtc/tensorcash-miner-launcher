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
if tensorcash_resolve_precision 12282 1 >/dev/null 2>&1; then
  echo 'FAIL: 12 GiB TP1 must not select FP8 without a serialized FP8 snapshot' >&2
  exit 1
fi
assert_eq fp8 "$(tensorcash_resolve_precision 16384 1)" '16 GiB auto profile'
assert_eq bf16 "$(tensorcash_resolve_precision 24564 1)" '24 GiB auto profile'
assert_eq fp8 "$(tensorcash_resolve_precision 6144 2)" '6 GiB FP8 TP2 profile'
assert_eq fp8 "$(tensorcash_resolve_precision 8192 2)" '8 GiB FP8 TP2 profile'
assert_eq bf16 "$(tensorcash_resolve_precision 11000 2)" 'legacy TP2 BF16 profile'
assert_eq bf16 "$(tensorcash_resolve_precision 8192 4)" 'legacy TP4 BF16 profile'

TENSORCASH_STATIC_FP8_TP1_AVAILABLE=true
assert_eq fp8 "$(tensorcash_resolve_precision 12282 1)" '12 GiB serialized FP8 auto profile'
assert_eq fp8 "$(tensorcash_resolve_precision 12000 1)" '12 GiB serialized FP8 lower boundary'
if tensorcash_resolve_precision 11999 1 >/dev/null 2>&1; then
  echo 'FAIL: serialized FP8 must retain its 12 GiB safety floor' >&2
  exit 1
fi

TENSORCASH_MODEL_PRECISION=fp8
assert_eq fp8 "$(tensorcash_resolve_precision 12282 1)" 'forced 12 GiB serialized FP8 profile'
assert_eq fp8 "$(tensorcash_resolve_precision 16384 1)" 'forced FP8 16 GiB profile'
TENSORCASH_MODEL_PRECISION=bf16
assert_eq bf16 "$(tensorcash_resolve_precision 12282 2)" 'forced TP2 BF16 profile'
if tensorcash_resolve_precision 12282 1 >/dev/null 2>&1; then
  echo 'FAIL: forced BF16 must reject a 12 GiB TP1 card' >&2
  exit 1
fi

unset TENSORCASH_MODEL_PRECISION
TENSORCASH_VLLM_QUANTIZATION=fp8
assert_eq fp8 "$(tensorcash_resolve_precision 24564 1)" 'legacy FP8 override'
unset TENSORCASH_VLLM_QUANTIZATION TENSORCASH_STATIC_FP8_TP1_AVAILABLE

echo 'runtime profile tests: OK'
