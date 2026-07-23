#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="${MINER_CONFIG:-$script_dir/miner.env}"
output_dir=""
copy_to=""
image_override=""

usage() {
  cat <<'EOF'
Usage:
  bash seed-export.sh [--output DIRECTORY] [--copy-to USER@HOST:PATH] [--image IMAGE]

Creates a source-free, checksum-protected TensorCash seed bundle from a host
that has already downloaded the runtime image and its usable model profile.
Both the canonical BF16 cache and the 12 GiB serialized-FP8 cache are
supported; the chain-pinned proof identity is retained in bundle.env.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

install_zstd_if_needed() {
  command -v zstd >/dev/null 2>&1 && return 0
  command -v apt-get >/dev/null 2>&1 || fail "zstd is required to create a seed bundle. Install it and run again."
  if ((EUID == 0)); then
    apt-get update
    apt-get install -y zstd
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y zstd
  else
    fail "zstd is missing and sudo is unavailable. Install zstd, then run again."
  fi
}

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$config" | tail -n 1
}

while (($#)); do
  case "$1" in
    --output) output_dir="${2:-}"; shift 2 ;;
    --copy-to) copy_to="${2:-}"; shift 2 ;;
    --image) image_override="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

require_command docker
require_command git
require_command tar
install_zstd_if_needed
require_command sha256sum
[[ -f "$config" ]] || fail "No miner.env at $config. Run start.sh successfully on this seed host first."
[[ -f "$script_dir/seed-install.sh" ]] || fail "seed-install.sh is missing from $script_dir. Update the launcher repository first."
git -C "$script_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "The launcher directory must be a Git checkout."

model_name="$(env_value MODEL_NAME)"
model_commit="$(env_value MODEL_COMMIT)"
pool_host="$(env_value POOL_HOST)"
pool_port="$(env_value POOL_PORT)"
source_image="${image_override:-$(env_value MINER_IMAGE)}"

[[ "$model_name" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "Invalid MODEL_NAME in miner.env."
[[ "$model_commit" =~ ^[0-9a-f]{40}$ ]] || fail "Invalid MODEL_COMMIT in miner.env."
[[ "$pool_host" =~ ^[A-Za-z0-9.-]+$ && "$pool_port" =~ ^[1-9][0-9]{0,4}$ ]] || fail "Invalid pool settings in miner.env."
[[ -n "$source_image" ]] || fail "MINER_IMAGE is missing from miner.env."

docker image inspect "$source_image" >/dev/null 2>&1 || fail "The configured image is not available locally: $source_image"

# A digest-only local configuration cannot be reproduced by docker save under
# the same digest syntax. Reuse one of its local canonical tags instead; a
# normal tag is retained exactly, including the separate Blackwell runtime.
bundle_image="$source_image"
if [[ "$bundle_image" == *@sha256:* ]]; then
  bundle_image="$(docker image inspect "$source_image" --format '{{range .RepoTags}}{{println .}}{{end}}' \
    | grep -E '^ghcr\.io/avalonbtc/tensorcash-miner:[A-Za-z0-9._-]+$' | head -n 1 || true)"
  [[ -n "$bundle_image" ]] || fail "The digest-only image has no local TensorCash tag. Re-run with --image ghcr.io/avalonbtc/tensorcash-miner:TAG."
fi
[[ "$bundle_image" =~ ^ghcr\.io/avalonbtc/tensorcash-miner:[A-Za-z0-9._-]+$ ]] || \
  fail "The bundle image must be a tagged Avalonbtc TensorCash runtime."

# A 12 GiB host intentionally has no canonical Qwen3-8B snapshot: it mines
# from the tested serialized FP8 artifact. Prefer the canonical cache when it
# exists, otherwise export that validated static artifact instead.
canonical_cache_name="${model_name//\//--}"
canonical_snapshot="$script_dir/runtime/models/hub/models--${canonical_cache_name}/snapshots/${model_commit}"
canonical_complete="$script_dir/runtime/models/.tensorcash-model-${canonical_cache_name}-${model_commit}.complete"
static_repository='Qwen/Qwen3-8B-FP8'
static_commit='220b46e3b2180893580a4454f21f22d3ebb187d3'
static_cache_name="${static_repository//\//--}"
static_snapshot="$script_dir/runtime/models/hub/models--${static_cache_name}/snapshots/${static_commit}"
static_complete="$static_snapshot/.tensorcash-static-fp8.complete"

if [[ -f "$canonical_snapshot/config.json" && -f "$canonical_complete" ]] && \
    compgen -G "$canonical_snapshot/*.safetensors" >/dev/null && \
    grep -Fqx "model=$model_name" "$canonical_complete" && \
    grep -Fqx "commit=$model_commit" "$canonical_complete"; then
  model_cache_kind='canonical'
  model_cache_repository="$model_name"
  model_cache_commit="$model_commit"
  model_cache_name="$canonical_cache_name"
elif [[ -f "$static_snapshot/config.json" && -f "$static_complete" ]] && \
    compgen -G "$static_snapshot/*.safetensors" >/dev/null && \
    grep -Fqx 'format=tensorcash-static-fp8-v1' "$static_complete" && \
    grep -Fqx "model=$model_name" "$static_complete" && \
    grep -Fqx "commit=$model_commit" "$static_complete" && \
    grep -Fqx "artifact_repository=$static_repository" "$static_complete" && \
    grep -Fqx "artifact_commit=$static_commit" "$static_complete" && \
    grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"fp8"' "$static_snapshot/config.json"; then
  model_cache_kind='serialized-fp8'
  model_cache_repository="$static_repository"
  model_cache_commit="$static_commit"
  model_cache_name="$static_cache_name"
else
  fail "No complete exportable model cache was found. Expected either $canonical_snapshot or the validated 12 GiB FP8 snapshot at $static_snapshot."
fi

bundle_version="${bundle_image##*:}"
bundle_label="${bundle_version}-${model_cache_kind}"
image_archive="tensorcash-image-${bundle_version}.tar.zst"
model_archive="tensorcash-model-${model_cache_name}.tar.zst"
launcher_archive='tensorcash-launcher.tar.zst'
image_id="$(docker image inspect "$source_image" --format '{{.Id}}')"
[[ "$image_id" =~ ^sha256:[A-Fa-f0-9]{64}$ ]] || fail "Could not read the local TensorCash image ID."
output_dir="${output_dir:-$HOME/tensorcash-seed-${bundle_label}}"
[[ ! -e "$output_dir" ]] || fail "Output already exists: $output_dir"

docker tag "$source_image" "$bundle_image"

mkdir -p "$output_dir"
chmod 700 "$output_dir"

cat > "$output_dir/bundle.env" <<EOF
BUNDLE_FORMAT=2
MINER_IMAGE=$bundle_image
IMAGE_ID=$image_id
IMAGE_ARCHIVE=$image_archive
MODEL_ARCHIVE=$model_archive
LAUNCHER_ARCHIVE=$launcher_archive
POOL_HOST=$pool_host
POOL_PORT=$pool_port
MODEL_NAME=$model_name
MODEL_COMMIT=$model_commit
MODEL_CACHE_KIND=$model_cache_kind
MODEL_CACHE_REPOSITORY=$model_cache_repository
MODEL_CACHE_COMMIT=$model_cache_commit
LAUNCHER_COMMIT=$(git -C "$script_dir" rev-parse HEAD)
CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cp "$script_dir/seed-install.sh" "$output_dir/seed-install.sh"
chmod 755 "$output_dir/seed-install.sh"

echo "Saving TensorCash runtime image..."
docker save "$bundle_image" | zstd -T0 -3 -o "$output_dir/$image_archive"

echo "Saving shared model cache..."
tar -C "$script_dir/runtime" -cf - models | zstd -T0 -3 -o "$output_dir/$model_archive"

echo "Saving public launcher files..."
git -C "$script_dir" archive --format=tar HEAD | zstd -T0 -3 -o "$output_dir/$launcher_archive"

(
  cd "$output_dir"
  sha256sum bundle.env seed-install.sh *.tar.zst > SHA256SUMS
)

echo "Seed bundle created: $output_dir"
(
  cd "$output_dir"
  ls -lh
)

if [[ -n "$copy_to" ]]; then
  require_command rsync
  destination="${copy_to%/}/$(basename "$output_dir")/"
  echo "Copying seed bundle with rsync resume support to $destination ..."
  echo "If the network disconnects, rerun the same command; completed bytes are verified and retained."
  rsync -a --info=progress2 --partial --append-verify "$output_dir/" "$destination"
  echo "Copy complete. On the destination, run: bash ${destination}seed-install.sh"
else
  echo "To copy it with resume support: rsync -a --info=progress2 --partial --append-verify '$output_dir/' root@TARGET:/root/$(basename "$output_dir")/"
fi
