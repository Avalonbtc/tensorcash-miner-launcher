#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$script_dir/native-vast.sh"

blackwell_ld_section="$(sed -n '/^native_runtime_ld_library_path()/,/^}/p' "$script")"
marker_section="$(sed -n '/^runtime_marker_is_current()/,/^}/p' "$script")"

grep -Fq 'blackwell_torch_library_path()' "$script" || {
  echo 'FAIL: Blackwell runtime must resolve the managed torch library path.' >&2
  exit 1
}
grep -Fq 'Path(torch.__file__).resolve().parent / "lib"' "$script" || {
  echo 'FAIL: Blackwell torch library lookup must come from the active venv.' >&2
  exit 1
}
grep -Fq '"$torch_library_path:$CUDA_HOME/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu"' <<<"$blackwell_ld_section" || {
  echo 'FAIL: managed torch libraries must precede host libraries at runtime.' >&2
  exit 1
}
grep -Fq 'LD_LIBRARY_PATH="$runtime_ld_library_path" "$NATIVE_PY"' <<<"$marker_section" || {
  echo 'FAIL: Blackwell marker probe must use the production linker path.' >&2
  exit 1
}

echo 'native Blackwell ABI linker tests: OK'
