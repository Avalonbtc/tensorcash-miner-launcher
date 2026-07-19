#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="${MINER_CONFIG:-$script_dir/miner.env}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

write_initial_config() {
  local payout="${PAYOUT_ACCOUNT:-}"
  local worker="${WORKER:-vast-$(hostname -s)}"
  local token

  [[ -n "$payout" ]] || fail "Set PAYOUT_ACCOUNT before the first run."
  [[ "$payout" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "PAYOUT_ACCOUNT contains unsupported characters."
  [[ "$worker" =~ ^[A-Za-z0-9._-]+$ ]] || fail "WORKER may use only letters, numbers, dot, underscore, and hyphen."

  if command -v openssl >/dev/null 2>&1; then
    token="$(openssl rand -hex 32)"
  else
    token="$(tr -d '-' </proc/sys/kernel/random/uuid)$(tr -d '-' </proc/sys/kernel/random/uuid)"
  fi

  umask 077
  cat > "$config" <<EOF
SIDECAR_IMAGE=ghcr.io/avalonbtc/tensortest-sidecar:0.1.0
MINER_IMAGE=ghcr.io/avalonbtc/tensortest-miner:0.1.0
POOL_HOST=${POOL_HOST:-119.91.239.215}
POOL_PORT=${POOL_PORT:-3336}
PAYOUT_ACCOUNT=$payout
WORKER=$worker
NOMP_SIDECAR_TOKEN=$token
MODEL_NAME=Qwen/Qwen3-0.6B
MODEL_COMMIT=c1899de289a04d12100db370d81485cdf75e47ca
MODEL_DIFFICULTY_NORMALIZER=1000000
MAX_MODEL_LEN=2048
VLLM_MAX_NUM_SEQS=1
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.78}
NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0,1,2,3}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-4}
MODELS_DATA=$script_dir/runtime/models
RUNTIME_DATA=$script_dir/runtime/data
EOF
}

if [[ ! -f "$config" ]]; then
  write_initial_config
fi

set -a
# shellcheck disable=SC1090
source "$config"
set +a

require_command nvidia-smi
require_command docker

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d '[:space:]')"
[[ "$gpu_count" -ge 4 ]] || fail "This test profile requires four GPUs in one Vast instance; detected $gpu_count."
[[ "$VLLM_TENSOR_PARALLEL_SIZE" == "4" ]] || fail "This image profile requires VLLM_TENSOR_PARALLEL_SIZE=4."
[[ "$NVIDIA_VISIBLE_DEVICES" == "0,1,2,3" ]] || fail "This image profile requires NVIDIA_VISIBLE_DEVICES=0,1,2,3."

if ! docker compose version >/dev/null 2>&1; then
  fail "Docker Compose v2 is required. Select a Vast template with Docker and NVIDIA Container Toolkit."
fi

if ! docker run --rm --gpus "device=0,1,2,3" nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L; then
  fail "Docker cannot access all four GPUs. Select a Vast template with NVIDIA Container Toolkit."
fi

mkdir -p "$MODELS_DATA" "$RUNTIME_DATA"
docker compose --env-file "$config" -f "$script_dir/docker-compose.yml" pull
docker compose --env-file "$config" -f "$script_dir/docker-compose.yml" up -d
docker compose --env-file "$config" -f "$script_dir/docker-compose.yml" logs -f miner
