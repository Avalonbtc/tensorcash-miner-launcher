#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="${MINER_CONFIG:-$script_dir/miner.env}"
pool_arg=""
wallet_arg=""
worker_arg=""
groups_arg=""
stop_only=false
update_only=false

usage() {
  cat <<'EOF'
Usage:
  bash start.sh --pool HOST:PORT --wallet PAYOUT --worker NAME [--gpu-groups '0,1;2,3']
  bash start.sh --update
  bash start.sh --stop
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

while (($#)); do
  case "$1" in
    --pool) pool_arg="${2:-}"; shift 2 ;;
    --wallet) wallet_arg="${2:-}"; shift 2 ;;
    --worker) worker_arg="${2:-}"; shift 2 ;;
    --gpu-groups) groups_arg="${2:-}"; shift 2 ;;
    --update) update_only=true; shift ;;
    --stop) stop_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

require_command docker

if ! docker compose version >/dev/null 2>&1; then
  fail "Docker Compose v2 is required. Use a GPU host with NVIDIA Container Toolkit."
fi

if "$stop_only"; then
  [[ -f "$config" ]] || exit 0
  set -a
  # shellcheck disable=SC1090
  source "$config"
  set +a
  IFS=';' read -r -a group_list <<< "${GPU_GROUPS:?GPU_GROUPS is missing from miner.env}"
  safe_worker="${WORKER//[^A-Za-z0-9_-]/-}"
  for index in "${!group_list[@]}"; do
    docker compose --project-name "tensorcash-${safe_worker}-g$((index + 1))" --env-file "$config" -f "$script_dir/docker-compose.yml" down --remove-orphans || true
  done
  exit 0
fi

if [[ ! -f "$config" ]]; then
  [[ -n "$pool_arg" && -n "$wallet_arg" && -n "$worker_arg" ]] || {
    usage
    fail "First launch requires --pool, --wallet, and --worker."
  }
  [[ "$pool_arg" =~ ^[A-Za-z0-9.-]+:[1-9][0-9]{0,4}$ ]] || fail "--pool must be HOST:PORT."
  [[ "$wallet_arg" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "--wallet contains unsupported characters."
  [[ "$worker_arg" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--worker contains unsupported characters."
  gpu_groups="${groups_arg:-0,1,2,3}"
  [[ "$gpu_groups" =~ ^[0-9]+(,[0-9]+)*(;[0-9]+(,[0-9]+)*)*$ ]] || fail "--gpu-groups must look like 0,1,2,3 or 0,1;2,3."
  token="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  umask 077
  cat > "$config" <<EOF
MINER_IMAGE=ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0
POOL_HOST=${pool_arg%:*}
POOL_PORT=${pool_arg##*:}
PAYOUT_ACCOUNT=$wallet_arg
WORKER=$worker_arg
NOMP_SIDECAR_TOKEN=$token
MODEL_NAME=Qwen/Qwen3-8B
MODEL_COMMIT=9c925d64d72725edaf899c6cb9c377fd0709d9c5
MODEL_DIFFICULTY_NORMALIZER=1000000
MAX_MODEL_LEN=2048
VLLM_MAX_NUM_SEQS=1
GPU_MEM_UTIL=0.78
GPU_GROUPS=$gpu_groups
MODELS_DATA=$script_dir/runtime/models
RUNTIME_DATA=$script_dir/runtime/data
TENSORCASH_POLL_MS=200
TENSORCASH_STATS_INTERVAL=30
TENSORCASH_SIDECAR_WAIT_SECONDS=1200
EOF
elif [[ -n "$pool_arg$wallet_arg$worker_arg$groups_arg" ]]; then
  fail "miner.env already exists; edit it explicitly or remove it before changing launch parameters."
fi

set -a
# shellcheck disable=SC1090
source "$config"
set +a

if "$update_only"; then
  update_image="${MINER_UPDATE_IMAGE:-ghcr.io/avalonbtc/tensorcash-miner:mainnet-latest}"
  echo "Checking for a new miner runtime: $update_image"
  docker pull "$update_image"
  pinned_image="$(docker image inspect "$update_image" --format '{{range .RepoDigests}}{{println .}}{{end}}' | grep '@sha256:' | head -n 1 || true)"
  [[ -n "$pinned_image" ]] || fail "The update image has no immutable registry digest."
  sed -i "s|^MINER_IMAGE=.*|MINER_IMAGE=$pinned_image|" "$config"
  MINER_IMAGE="$pinned_image"
  export MINER_IMAGE
  echo "Pinned this host to $MINER_IMAGE"
fi

require_command nvidia-smi
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || fail "No NVIDIA GPUs are visible on this host."
[[ "$POOL_HOST" =~ ^[A-Za-z0-9.-]+$ && "$POOL_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || fail "Invalid pool settings in miner.env."
[[ "$PAYOUT_ACCOUNT" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "Invalid payout account in miner.env."
[[ "$WORKER" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Invalid worker in miner.env."
[[ "$NOMP_SIDECAR_TOKEN" =~ ^[A-Fa-f0-9]{32,}$ ]] || fail "Invalid sidecar token in miner.env."
[[ "$GPU_GROUPS" =~ ^[0-9]+(,[0-9]+)*(;[0-9]+(,[0-9]+)*)*$ ]] || fail "Invalid GPU_GROUPS in miner.env."

IFS=';' read -r -a group_list <<< "$GPU_GROUPS"
declare -A seen_gpu=()
for group in "${group_list[@]}"; do
  IFS=',' read -r -a group_gpus <<< "$group"
  for gpu in "${group_gpus[@]}"; do
    [[ "$gpu" -lt "$gpu_count" ]] || fail "GPU $gpu does not exist; host exposes $gpu_count GPU(s)."
    [[ -z "${seen_gpu[$gpu]:-}" ]] || fail "GPU $gpu appears in more than one group."
    seen_gpu[$gpu]=1
  done
done

mkdir -p "$MODELS_DATA" "$RUNTIME_DATA"
chmod 700 "$MODELS_DATA" "$RUNTIME_DATA"
docker compose --env-file "$config" -f "$script_dir/docker-compose.yml" pull

model_cache_name="${MODEL_NAME//\//--}"
model_config="$MODELS_DATA/hub/models--${model_cache_name}/snapshots/${MODEL_COMMIT}/config.json"
if [[ ! -f "$model_config" ]]; then
  echo "Downloading ${MODEL_NAME}@${MODEL_COMMIT} once into $MODELS_DATA ..."
  docker run --rm --entrypoint python3 \
    -e MODEL_NAME -e MODEL_COMMIT \
    -v "$MODELS_DATA:/models" \
    "$MINER_IMAGE" \
    -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["MODEL_NAME"], revision=os.environ["MODEL_COMMIT"], cache_dir="/models/hub")'
fi
[[ -f "$model_config" ]] || fail "Model download finished without the expected pinned snapshot."

safe_worker="${WORKER//[^A-Za-z0-9_-]/-}"
for index in "${!group_list[@]}"; do
  group_number=$((index + 1))
  group="${group_list[$index]}"
  IFS=',' read -r -a group_gpus <<< "$group"
  group_worker="${WORKER}-g${group_number}"
  group_runtime="$RUNTIME_DATA/group-${group_number}"
  mkdir -p "$group_runtime"
  echo "Starting ${group_worker}: GPUs ${group}, TP=${#group_gpus[@]}"
  (
    export WORKER="$group_worker"
    export NVIDIA_VISIBLE_DEVICES="$group"
    export VLLM_TENSOR_PARALLEL_SIZE="${#group_gpus[@]}"
    export RUNTIME_DATA="$group_runtime"
    docker compose --project-name "tensorcash-${safe_worker}-g${group_number}" --env-file "$config" -f "$script_dir/docker-compose.yml" up -d --remove-orphans
  )
done

echo "TensorCash started. Model cache: $MODELS_DATA"
echo "Use: docker ps --filter 'name=tensorcash-'"
