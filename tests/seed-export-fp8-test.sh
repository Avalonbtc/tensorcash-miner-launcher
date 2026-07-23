#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

launcher_dir="$tmp_dir/launcher"
mock_bin="$tmp_dir/bin"
mkdir -p "$launcher_dir/runtime/models" "$mock_bin" "$tmp_dir/home"
cp "$repo_dir/seed-export.sh" "$repo_dir/seed-install.sh" "$launcher_dir/"
chmod +x "$launcher_dir/seed-export.sh" "$launcher_dir/seed-install.sh"

cat > "$launcher_dir/miner.env" <<'EOF'
MINER_IMAGE=ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0
POOL_HOST=pool.example.test
POOL_PORT=3336
MODEL_NAME=Qwen/Qwen3-8B
MODEL_COMMIT=9c925d64d72725edaf899c6cb9c377fd0709d9c5
EOF

snapshot="$launcher_dir/runtime/models/hub/models--Qwen--Qwen3-8B-FP8/snapshots/220b46e3b2180893580a4454f21f22d3ebb187d3"
mkdir -p "$snapshot"
printf '{"quant_method":"fp8"}\n' > "$snapshot/config.json"
printf 'weights\n' > "$snapshot/model-00001-of-00001.safetensors"
cat > "$snapshot/.tensorcash-static-fp8.complete" <<'EOF'
format=tensorcash-static-fp8-v1
model=Qwen/Qwen3-8B
commit=9c925d64d72725edaf899c6cb9c377fd0709d9c5
artifact_repository=Qwen/Qwen3-8B-FP8
artifact_commit=220b46e3b2180893580a4454f21f22d3ebb187d3
EOF

cat > "$mock_bin/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1 ${2:-}" in
  'image inspect')
    case "$*" in
      *'{{.Id}}'*) echo 'sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef' ;;
      *'{{range .RepoTags}}'*) echo 'ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0' ;;
      *) exit 0 ;;
    esac
    ;;
  tag\ *) exit 0 ;;
  save\ *) printf 'fake TensorCash image\n' ;;
  *) echo "unexpected docker invocation: $*" >&2; exit 64 ;;
esac
EOF

cat > "$mock_bin/zstd" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
output=''
while (($#)); do
  case "$1" in
    -o) output="$2"; shift 2 ;;
    -T0|-3) shift ;;
    *) echo "unexpected zstd argument: $1" >&2; exit 64 ;;
  esac
done
[[ -n "$output" ]] || { echo 'missing zstd output path' >&2; exit 64; }
cat > "$output"
EOF

cat > "$mock_bin/git" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$*" in
  *' rev-parse --is-inside-work-tree') echo true ;;
  *' rev-parse HEAD') echo '0123456789abcdef0123456789abcdef01234567' ;;
  *' archive --format=tar HEAD') printf 'fake public launcher archive\n' ;;
  *) echo "unexpected git invocation: $*" >&2; exit 64 ;;
esac
EOF
chmod +x "$mock_bin/docker" "$mock_bin/zstd" "$mock_bin/git"

output_dir="$tmp_dir/seed"
PATH="$mock_bin:$PATH" HOME="$tmp_dir/home" \
  bash "$launcher_dir/seed-export.sh" --output "$output_dir" >/dev/null

# shellcheck disable=SC1090
source "$output_dir/bundle.env"
[[ "$BUNDLE_FORMAT" == 2 ]]
[[ "$MODEL_CACHE_KIND" == serialized-fp8 ]]
[[ "$MODEL_CACHE_REPOSITORY" == Qwen/Qwen3-8B-FP8 ]]
[[ "$MODEL_CACHE_COMMIT" == 220b46e3b2180893580a4454f21f22d3ebb187d3 ]]
[[ "$MODEL_ARCHIVE" == tensorcash-model-Qwen--Qwen3-8B-FP8.tar.zst ]]
[[ -s "$output_dir/$IMAGE_ARCHIVE" && -s "$output_dir/$MODEL_ARCHIVE" ]]
grep -Fqx "$MODEL_ARCHIVE" < <(awk '{print $2}' "$output_dir/SHA256SUMS")

echo 'seed export serialized FP8 test: OK'
