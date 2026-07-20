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
that has already downloaded the runtime image and chain-pinned model.
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

model_cache_name="${model_name//\//--}"
model_config="$script_dir/runtime/models/hub/models--${model_cache_name}/snapshots/${model_commit}/config.json"
model_complete="$script_dir/runtime/models/.tensorcash-model-${model_cache_name}-${model_commit}.complete"
[[ -f "$model_config" ]] || fail "Pinned model snapshot is missing: $model_config"
[[ -f "$model_complete" ]] || fail "Pinned model cache has no completion marker. Update the launcher and run bash start.sh once on this seed host before exporting."

bundle_version="mainnet-0.1.0"
bundle_image="ghcr.io/avalonbtc/tensorcash-miner:${bundle_version}"
output_dir="${output_dir:-$HOME/tensorcash-seed-${bundle_version}}"
[[ ! -e "$output_dir" ]] || fail "Output already exists: $output_dir"

docker image inspect "$source_image" >/dev/null 2>&1 || fail "The configured image is not available locally: $source_image"
docker tag "$source_image" "$bundle_image"

mkdir -p "$output_dir"
chmod 700 "$output_dir"

cat > "$output_dir/bundle.env" <<EOF
BUNDLE_FORMAT=1
MINER_IMAGE=$bundle_image
POOL_HOST=$pool_host
POOL_PORT=$pool_port
MODEL_NAME=$model_name
MODEL_COMMIT=$model_commit
LAUNCHER_COMMIT=$(git -C "$script_dir" rev-parse HEAD)
CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

cp "$script_dir/seed-install.sh" "$output_dir/seed-install.sh"
chmod 755 "$output_dir/seed-install.sh"

echo "Saving TensorCash runtime image..."
docker save "$bundle_image" | zstd -T0 -3 -o "$output_dir/tensorcash-image-${bundle_version}.tar.zst"

echo "Saving shared model cache..."
tar -C "$script_dir/runtime" -cf - models | zstd -T0 -3 -o "$output_dir/tensorcash-model-${model_cache_name}.tar.zst"

echo "Saving public launcher files..."
git -C "$script_dir" archive --format=tar HEAD | zstd -T0 -3 -o "$output_dir/tensorcash-launcher.tar.zst"

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
