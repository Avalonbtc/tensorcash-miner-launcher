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
payout account and optional worker name. The first launch is fully offline:
the bundled image and model cache are used without a registry pull or Git sync.
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

validate_model_cache() {
  local kind="$1" repository="$2" commit="$3" cache_name snapshot marker
  cache_name="${repository//\//--}"
  snapshot="$install_dir/runtime/models/hub/models--${cache_name}/snapshots/${commit}"
  [[ -f "$snapshot/config.json" ]] || fail "Bundled model snapshot is missing: $snapshot/config.json"
  compgen -G "$snapshot/*.safetensors" >/dev/null || fail "Bundled model snapshot has no safetensors weights: $snapshot"

  case "$kind" in
    canonical)
      [[ "$repository" == "$model_name" && "$commit" == "$model_commit" ]] || \
        fail "Canonical bundle metadata does not match the chain-pinned model."
      marker="$install_dir/runtime/models/.tensorcash-model-${cache_name}-${commit}.complete"
      [[ -f "$marker" ]] || fail "Bundled canonical model has no completion marker: $marker"
      grep -Fqx "model=$model_name" "$marker" && grep -Fqx "commit=$model_commit" "$marker" || \
        fail "Bundled canonical completion marker does not match the chain-pinned model."
      ;;
    serialized-fp8)
      [[ "$repository" == 'Qwen/Qwen3-8B-FP8' && "$commit" == '220b46e3b2180893580a4454f21f22d3ebb187d3' ]] || \
        fail "Serialized FP8 bundle metadata is not the tested TensorCash artifact."
      marker="$snapshot/.tensorcash-static-fp8.complete"
      [[ -f "$marker" ]] || fail "Bundled serialized FP8 model has no attestation: $marker"
      grep -Fqx 'format=tensorcash-static-fp8-v1' "$marker" && \
      grep -Fqx "model=$model_name" "$marker" && \
      grep -Fqx "commit=$model_commit" "$marker" && \
      grep -Fqx "artifact_repository=$repository" "$marker" && \
      grep -Fqx "artifact_commit=$commit" "$marker" || \
        fail "Bundled serialized FP8 attestation does not match the TensorCash model."
      grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"fp8"' "$snapshot/config.json" || \
        fail "Bundled serialized FP8 model lacks its FP8 quantization configuration."
      ;;
    *) fail "Unsupported bundled model cache kind: $kind" ;;
  esac
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
bundle_format="$(manifest_value BUNDLE_FORMAT)"
default_pool_host="$(manifest_value POOL_HOST)"
default_pool_port="$(manifest_value POOL_PORT)"
model_name="$(manifest_value MODEL_NAME)"
model_commit="$(manifest_value MODEL_COMMIT)"

[[ "$image" =~ ^ghcr\.io/avalonbtc/tensorcash-miner:[A-Za-z0-9._-]+$ ]] || fail "Bundle image name is invalid."
[[ "$default_pool_host" =~ ^[A-Za-z0-9.-]+$ && "$default_pool_port" =~ ^[1-9][0-9]{0,4}$ ]] || fail "Bundle pool setting is invalid."
[[ "$model_name" == Qwen/Qwen3-8B && "$model_commit" =~ ^[0-9a-f]{40}$ ]] || fail "This installer only accepts the chain-pinned Qwen3-8B profile."
[[ "$groups_arg" == auto || "$groups_arg" =~ ^[0-9]+(,[0-9]+)*(\;[0-9]+(,[0-9]+)*)*$ ]] || fail "--gpu-groups must be auto or look like 0,1;2,3."

case "$bundle_format" in
  1)
    # Compatibility with seed bundles exported before serialized FP8 support.
    bundle_version="${image##*:}"
    model_cache_repository="$model_name"
    model_cache_commit="$model_commit"
    model_cache_kind='canonical'
    image_archive="tensorcash-image-${bundle_version}.tar.zst"
    model_archive="tensorcash-model-${model_cache_repository//\//--}.tar.zst"
    launcher_archive='tensorcash-launcher.tar.zst'
    expected_image_id=''
    ;;
  2)
    expected_image_id="$(manifest_value IMAGE_ID)"
    image_archive="$(manifest_value IMAGE_ARCHIVE)"
    model_archive="$(manifest_value MODEL_ARCHIVE)"
    launcher_archive="$(manifest_value LAUNCHER_ARCHIVE)"
    model_cache_kind="$(manifest_value MODEL_CACHE_KIND)"
    model_cache_repository="$(manifest_value MODEL_CACHE_REPOSITORY)"
    model_cache_commit="$(manifest_value MODEL_CACHE_COMMIT)"
    [[ "$expected_image_id" =~ ^sha256:[A-Fa-f0-9]{64}$ ]] || fail "Bundle image ID is invalid."
    [[ "$image_archive" =~ ^tensorcash-image-[A-Za-z0-9._-]+\.tar\.zst$ ]] || fail "Bundle image archive name is invalid."
    [[ "$model_archive" =~ ^tensorcash-model-[A-Za-z0-9._-]+\.tar\.zst$ ]] || fail "Bundle model archive name is invalid."
    [[ "$launcher_archive" == 'tensorcash-launcher.tar.zst' ]] || fail "Bundle launcher archive name is invalid."
    [[ "$model_cache_repository" =~ ^[A-Za-z0-9._/-]+$ && "$model_cache_commit" =~ ^[0-9a-f]{40}$ ]] || \
      fail "Bundle model-cache metadata is invalid."
    ;;
  *) fail "Unsupported TensorCash seed bundle format: $bundle_format" ;;
esac

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

image_archive="$bundle_dir/$image_archive"
model_archive="$bundle_dir/$model_archive"
launcher_archive="$bundle_dir/$launcher_archive"
[[ -f "$image_archive" && -f "$model_archive" && -f "$launcher_archive" ]] || fail "One or more expected bundle archives are missing."

docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required."
nvidia-smi --query-gpu=index --format=csv,noheader | grep -q . || fail "No NVIDIA GPUs are visible on this host."

echo "Loading TensorCash runtime image..."
zstd -dc "$image_archive" | docker load
docker image inspect "$image" >/dev/null 2>&1 || fail "The loaded image does not contain the expected tag: $image"
if [[ -n "$expected_image_id" ]]; then
  loaded_image_id="$(docker image inspect "$image" --format '{{.Id}}')"
  [[ "$loaded_image_id" == "$expected_image_id" ]] || \
    fail "The loaded image ID does not match the verified bundle manifest."
fi

mkdir -p "$install_dir/runtime"
echo "Installing public launcher files..."
zstd -dc "$launcher_archive" | tar -C "$install_dir" -xpf -
echo "Installing shared model cache..."
zstd -dc "$model_archive" | tar -C "$install_dir/runtime" -xpf -
find "$install_dir/runtime/models" -type d -exec chmod a+rx {} +
find "$install_dir/runtime/models" -type f -exec chmod a+r {} +
validate_model_cache "$model_cache_kind" "$model_cache_repository" "$model_cache_commit"

echo "Starting TensorCash miner..."
cd "$install_dir"
# The bundle is intentionally source-free (`git archive` does not carry .git).
# Skip only this first self-update so an air-gapped destination can install;
# the generated miner.env re-enables normal launcher updates for later starts.
TENSORCASH_AUTO_UPDATE=false TENSORCASH_SKIP_IMAGE_PULL=1 \
  TENSORCASH_INITIAL_MINER_IMAGE="$image" bash start.sh \
  --pool "$pool" \
  --wallet "$wallet_arg" \
  --worker "$worker_arg" \
  --gpu-groups "$groups_arg"

echo "TensorCash started. Follow logs with: docker ps --filter 'name=tensorcash-'"
