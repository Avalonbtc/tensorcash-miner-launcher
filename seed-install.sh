#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
install_dir="$HOME/tensorcash-miner"
pool_arg=""
wallet_arg=""
worker_arg=""
groups_arg="auto"

usage() {
  cat <<'EOF'
Usage:
  bash seed-install.sh [--install-dir DIRECTORY] [--pool HOST:PORT] [--wallet PAYOUT] [--worker NAME] [--gpu-groups auto|GROUPS]

With no options, the bundled pool is used and the script prompts only for the
payout account and optional worker name.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

manifest_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "$bundle_dir/bundle.env" | tail -n 1
}

install_zstd_if_needed() {
  command -v zstd >/dev/null 2>&1 && return 0
  command -v apt-get >/dev/null 2>&1 || fail "zstd is required to unpack this bundle. Install it and run again."
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

while (($#)); do
  case "$1" in
    --install-dir) install_dir="${2:-}"; shift 2 ;;
    --pool) pool_arg="${2:-}"; shift 2 ;;
    --wallet) wallet_arg="${2:-}"; shift 2 ;;
    --worker) worker_arg="${2:-}"; shift 2 ;;
    --gpu-groups) groups_arg="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

[[ -f "$bundle_dir/bundle.env" && -f "$bundle_dir/SHA256SUMS" ]] || fail "This directory is not a complete TensorCash seed bundle."
install_zstd_if_needed
require_command sha256sum
require_command tar
require_command docker
require_command nvidia-smi

(
  cd "$bundle_dir"
  sha256sum -c SHA256SUMS
)

image="$(manifest_value MINER_IMAGE)"
default_pool_host="$(manifest_value POOL_HOST)"
default_pool_port="$(manifest_value POOL_PORT)"
model_name="$(manifest_value MODEL_NAME)"
model_commit="$(manifest_value MODEL_COMMIT)"

[[ "$image" == ghcr.io/avalonbtc/tensorcash-miner:* ]] || fail "Bundle image name is invalid."
[[ "$default_pool_host" =~ ^[A-Za-z0-9.-]+$ && "$default_pool_port" =~ ^[1-9][0-9]{0,4}$ ]] || fail "Bundle pool setting is invalid."
[[ "$model_name" == Qwen/Qwen3-8B && "$model_commit" =~ ^[0-9a-f]{40}$ ]] || fail "This installer only accepts the chain-pinned Qwen3-8B profile."
[[ "$groups_arg" == auto || "$groups_arg" =~ ^[0-9]+(,[0-9]+)*(\;[0-9]+(,[0-9]+)*)*$ ]] || fail "--gpu-groups must be auto or look like 0,1;2,3."

pool="${pool_arg:-${default_pool_host}:${default_pool_port}}"
[[ "$pool" =~ ^[A-Za-z0-9.-]+:[1-9][0-9]{0,4}$ ]] || fail "--pool must be HOST:PORT."

if [[ -z "$wallet_arg" ]]; then
  read -r -p "Payout account: " wallet_arg
fi
[[ "$wallet_arg" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "The payout account is empty or contains unsupported characters."

default_worker="tc-$(hostname | tr -c 'A-Za-z0-9_-' '-')"
if [[ -z "$worker_arg" ]]; then
  read -r -p "Worker name [$default_worker]: " worker_arg
  worker_arg="${worker_arg:-$default_worker}"
fi
[[ "$worker_arg" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Worker name contains unsupported characters."
[[ ! -e "$install_dir" ]] || fail "Install directory already exists: $install_dir. Use --install-dir for a new directory; never overwrite another miner's miner.env."

bundle_version="${image##*:}"
model_cache_name="${model_name//\//--}"
image_archive="$bundle_dir/tensorcash-image-${bundle_version}.tar.zst"
model_archive="$bundle_dir/tensorcash-model-${model_cache_name}.tar.zst"
launcher_archive="$bundle_dir/tensorcash-launcher.tar.zst"
[[ -f "$image_archive" && -f "$model_archive" && -f "$launcher_archive" ]] || fail "One or more expected bundle archives are missing."

docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
nvidia-smi --query-gpu=index --format=csv,noheader | grep -q . || fail "No NVIDIA GPUs are visible on this host."

echo "Loading TensorCash runtime image..."
zstd -dc "$image_archive" | docker load
docker image inspect "$image" >/dev/null 2>&1 || fail "The loaded image does not contain the expected tag: $image"

mkdir -p "$install_dir/runtime"
echo "Installing public launcher files..."
zstd -dc "$launcher_archive" | tar -C "$install_dir" -xpf -
echo "Installing shared model cache..."
zstd -dc "$model_archive" | tar -C "$install_dir/runtime" -xpf -

echo "Starting TensorCash miner..."
cd "$install_dir"
TENSORCASH_SKIP_IMAGE_PULL=1 bash start.sh \
  --pool "$pool" \
  --wallet "$wallet_arg" \
  --worker "$worker_arg" \
  --gpu-groups "$groups_arg"

echo "TensorCash started. Follow logs with: docker ps --filter 'name=tensorcash-'"
