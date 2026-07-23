#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for launcher in start.sh native-vast.sh; do
  grep -Fq 'tensorcash_compute_capability_is_pre_fp8' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must route pre-SM80 GPUs away from FP8." >&2
    exit 1
  }
  grep -Fq 'FP16' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must explain the pre-SM80 FP16 profile." >&2
    exit 1
  }
done

grep -Fq 'fp16) args+=(--dtype float16)' "$script_dir/vllm-local-cache.sh" || {
  echo 'FAIL: vLLM launcher must use explicit FP16 for the legacy profile.' >&2
  exit 1
}

echo 'legacy GPU profile tests: OK'
