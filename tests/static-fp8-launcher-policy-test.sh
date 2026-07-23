#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for launcher in start.sh native-vast.sh; do
  grep -Fq 'tensorcash_static_fp8_tp1_download_needed "$memory"' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must use the shared serialized-FP8 download policy." >&2
    exit 1
  }
  grep -Fq '12--21.9 GiB' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must document its 12--21.9 GiB serialized-FP8 profile." >&2
    exit 1
  }
done

grep -Fq 'tensorcash_can_use_static_fp8_tp1 "$min_memory"' "$script_dir/start.sh" || {
  echo 'FAIL: Docker group resolution must select the validated static FP8 snapshot.' >&2
  exit 1
}
grep -Fq 'tensorcash_can_use_static_fp8_tp1 "$min_memory"' "$script_dir/native-vast.sh" || {
  echo 'FAIL: native group resolution must select the validated static FP8 snapshot.' >&2
  exit 1
}

echo 'serialized FP8 launcher policy tests: OK'
