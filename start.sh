#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="${MINER_CONFIG:-$script_dir/miner.env}"
# shellcheck source=launcher-sync.sh
source "$script_dir/launcher-sync.sh"
if [[ "${TENSORCASH_LAUNCHER_REEXECUTED:-0}" != 1 ]] && launcher_command_starts_runtime "$@"; then
  if launcher_auto_update_enabled "$config"; then
    launcher_sync_latest "$script_dir" || {
      echo "ERROR: forced TensorCash launcher update failed; set TENSORCASH_AUTO_UPDATE=false only for an emergency offline recovery." >&2
      exit 2
    }
    export TENSORCASH_LAUNCHER_REEXECUTED=1
    exec bash "$script_dir/start.sh" "$@"
  else
    auto_update_status=$?
    if (( auto_update_status != 1 )); then
      echo "ERROR: invalid TENSORCASH_AUTO_UPDATE setting." >&2
      exit 2
    fi
  fi
fi
# shellcheck source=runtime-profile.sh
source "$script_dir/runtime-profile.sh"
pool_arg=""
wallet_arg=""
worker_arg=""
groups_arg=""
stop_only=false
update_only=false

readonly LEGACY_RUNTIME_IMAGE='ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0'
readonly BLACKWELL_RUNTIME_IMAGE='ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.1-blackwell'

usage() {
  cat <<'EOF'
Usage:
  bash start.sh --pool HOST:PORT --wallet PAYOUT --worker NAME [--gpu-groups auto|GROUPS]
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

has_blackwell_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  # CUDA capability is more reliable than product marketing names and covers
  # RTX 50-series as well as future Blackwell workstation cards.  Keep a name
  # fallback for older nvidia-smi releases that do not expose compute_cap.
  if nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
      | awk '{ if (($1 + 0) >= 12) found = 1 } END { exit !found }'; then
    return 0
  fi
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
    | grep -Eqi '(RTX 50[0-9]0|Blackwell|GB10)'
}

default_runtime_image() {
  # Keep a newly-created config on the known legacy tag until the separate
  # Blackwell image has actually been published. The migration below checks
  # the registry before changing it; this avoids persisting a manifest-unknown
  # tag after a failed CI image build.
  printf '%s\n' "$LEGACY_RUNTIME_IMAGE"
}

blackwell_runtime_published() {
  command -v docker >/dev/null 2>&1 || return 1
  docker manifest inspect "$BLACKWELL_RUNTIME_IMAGE" >/dev/null 2>&1
}

require_published_blackwell_runtime() {
  blackwell_runtime_published && return 0
  fail "RTX 50-series GPU detected, but $BLACKWELL_RUNTIME_IMAGE is not published in GHCR yet. Docker mining is unavailable for this GPU until the Blackwell image build succeeds; use native-vast.sh instead."
}

positive_integer() {
  local value="$1" name="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer."
}

pull_image_with_retries() {
  local image="$1"
  local attempts="${TENSORCASH_IMAGE_PULL_ATTEMPTS:-12}"
  local delay="${TENSORCASH_IMAGE_PULL_DELAY_SECONDS:-15}"
  local attempt

  positive_integer "$attempts" TENSORCASH_IMAGE_PULL_ATTEMPTS
  positive_integer "$delay" TENSORCASH_IMAGE_PULL_DELAY_SECONDS

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    echo "Pulling TensorCash runtime image (attempt $attempt/$attempts): $image"
    if docker pull "$image"; then
      return 0
    fi
    if ((attempt < attempts)); then
      echo "Image pull interrupted. Docker keeps completed layers; retrying in ${delay}s..." >&2
      sleep "$delay"
    fi
  done

  fail "Could not pull $image after $attempts attempts. Set TENSORCASH_IMAGE_ARCHIVE_URL to a resumable seed archive, or configure the Docker daemon proxy with docker-proxy.sh."
}

load_image_archive() {
  local image="$1"
  local url="${TENSORCASH_IMAGE_ARCHIVE_URL:-}"
  local expected_sha256="${TENSORCASH_IMAGE_ARCHIVE_SHA256:-}"
  local archive_dir archive_name archive_file partial_file
  local -a curl_proxy=() curl_retry_all=()

  [[ -n "$url" ]] || return 1
  [[ "$url" =~ ^https?:// ]] || fail "TENSORCASH_IMAGE_ARCHIVE_URL must be an HTTP(S) URL."
  if [[ -n "$expected_sha256" ]]; then
    [[ "$expected_sha256" =~ ^[A-Fa-f0-9]{64}$ ]] || fail "TENSORCASH_IMAGE_ARCHIVE_SHA256 must be a SHA-256 hex digest."
  fi

  require_command curl
  require_command zstd
  require_command sha256sum

  archive_dir="${TENSORCASH_IMAGE_ARCHIVE_CACHE_DIR:-$script_dir/runtime/image-download}"
  archive_name="${image//[^A-Za-z0-9._-]/-}.tar.zst"
  archive_file="$archive_dir/$archive_name"
  partial_file="$archive_file.partial"
  mkdir -p "$archive_dir"
  chmod 700 "$archive_dir"

  if [[ -n "${TENSORCASH_HTTP_PROXY:-}" ]]; then
    curl_proxy=(--proxy "$TENSORCASH_HTTP_PROXY")
  fi
  if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    curl_retry_all=(--retry-all-errors)
  fi

  if [[ ! -f "$archive_file" ]]; then
    echo "Fetching resumable TensorCash image archive from $url"
    echo "The .partial file is retained on interruption; rerun start.sh to continue."
    curl --fail --location --continue-at - --retry 8 "${curl_retry_all[@]}" \
      --connect-timeout 30 --speed-time 90 --speed-limit 10240 \
      "${curl_proxy[@]}" --output "$partial_file" "$url"

    if [[ -n "$expected_sha256" ]]; then
      printf '%s  %s\n' "$expected_sha256" "$partial_file" | sha256sum -c -
    fi
    mv "$partial_file" "$archive_file"
  fi

  if [[ -n "$expected_sha256" ]]; then
    printf '%s  %s\n' "$expected_sha256" "$archive_file" | sha256sum -c -
  fi

  echo "Loading TensorCash runtime image archive..."
  zstd -dc "$archive_file" | docker load
  docker image inspect "$image" >/dev/null 2>&1 || fail "Archive loaded, but does not contain the configured image: $image"
}

ensure_runtime_image() {
  local image="$1"
  if docker image inspect "$image" >/dev/null 2>&1; then
    echo "Using already-loaded TensorCash runtime image: $image"
    return 0
  fi

  if [[ -n "${TENSORCASH_IMAGE_ARCHIVE_URL:-}" ]]; then
    load_image_archive "$image"
  else
    pull_image_with_retries "$image"
  fi
}

ensure_compatible_miner_binary() {
  local binary_dir="$script_dir/runtime/bin"
  local binary_path="$binary_dir/niuquanminer"
  # v9 keeps the canonical model, proof bytes, pool target, and consensus
  # unchanged while updating the controller's runtime scheduling behavior.
  # The controller is glibc-2.35-compatible for Ubuntu 22.04/HiveOS hosts.
  local binary_url="${TENSORCASH_CONTROLLER_URL:-https://github.com/Avalonbtc/tensorcash-miner-launcher/releases/download/controller-glibc235-v9/niuquanminer-linux-amd64-glibc235}"
  local expected_sha256="${TENSORCASH_CONTROLLER_SHA256:-88fad32fd31782bb9f9dd6ddca516ef7dfb023ecdc349949ac976c3428220c4a}"
  local temp_path
  local -a proxy_args=() retry_args=()

  require_command curl
  require_command sha256sum
  [[ "$binary_url" =~ ^https?:// ]] || fail "TENSORCASH_CONTROLLER_URL must be an HTTP(S) URL."
  [[ "$expected_sha256" =~ ^[A-Fa-f0-9]{64}$ ]] || fail "TENSORCASH_CONTROLLER_SHA256 must be a SHA-256 hex digest."
  mkdir -p "$binary_dir"
  chmod 700 "$binary_dir"
  if [[ -x "$binary_path" ]] && printf '%s  %s\n' "$expected_sha256" "$binary_path" | sha256sum -c - >/dev/null 2>&1; then
    MINER_BINARY_PATH="$binary_path"
    export MINER_BINARY_PATH
    echo "Using verified TensorCash controller: $binary_path"
    return 0
  fi

  [[ -n "${TENSORCASH_HTTP_PROXY:-}" ]] && proxy_args=(--proxy "$TENSORCASH_HTTP_PROXY")
  if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    retry_args=(--retry-all-errors)
  fi
  temp_path="$binary_dir/.niuquanminer.$$"
  rm -f "$temp_path"
  echo "Downloading glibc-compatible TensorCash controller (about 3 MB)..."
  if ! curl --fail --location --retry 8 "${retry_args[@]}" --connect-timeout 30 \
    "${proxy_args[@]}" --output "$temp_path" "$binary_url"; then
    rm -f "$temp_path"
    fail "Could not download the compatible TensorCash controller."
  fi
  if ! printf '%s  %s\n' "$expected_sha256" "$temp_path" | sha256sum -c -; then
    rm -f "$temp_path"
    fail "Compatible TensorCash controller checksum mismatch."
  fi
  chmod 755 "$temp_path"
  mv -f "$temp_path" "$binary_path"
  MINER_BINARY_PATH="$binary_path"
  export MINER_BINARY_PATH
  echo "Installed verified glibc-compatible TensorCash controller."
}

download_model_with_retries() {
  local model_name="$1" model_commit="$2" models_data="$3"
  local attempts="${TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS:-12}"
  local delay="${TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS:-15}"
  local attempt proxy_var proxy_value
  local -a proxy_env=()

  positive_integer "$attempts" TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS
  positive_integer "$delay" TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS

  for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
    proxy_value="${!proxy_var:-}"
    [[ -n "$proxy_value" ]] && proxy_env+=(-e "$proxy_var=$proxy_value")
  done
  if [[ -n "${TENSORCASH_HTTP_PROXY:-}" ]]; then
    proxy_env+=(-e "HTTP_PROXY=$TENSORCASH_HTTP_PROXY" -e "HTTPS_PROXY=$TENSORCASH_HTTP_PROXY")
  fi

  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    echo "Downloading ${model_name}@${model_commit} (attempt $attempt/$attempts; existing cache is reused)..."
    if docker run --rm --entrypoint python3 \
      -e MODEL_NAME -e MODEL_COMMIT \
      "${proxy_env[@]}" \
      -v "$models_data:/models" \
      "$MINER_IMAGE" \
      -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["MODEL_NAME"], revision=os.environ["MODEL_COMMIT"], cache_dir="/models/hub")'; then
      return 0
    fi
    if ((attempt < attempts)); then
      echo "Model download interrupted. Retrying from the shared cache in ${delay}s..." >&2
      sleep "$delay"
    fi
  done

  fail "Could not download the pinned model after $attempts attempts. Use seed-export.sh plus rsync for a resumable offline transfer."
}

auto_gpu_groups() {
  local bf16_min fp8_min fp8_tp2_min tp2_min tp4_min
  local -a memories=() singles=() fp8_tp2=() tp2=() tp4=() groups=() leftovers=() detected=()
  local index memory start precision_mode
  tensorcash_validate_vram_thresholds || fail "Invalid TensorCash VRAM threshold configuration."
  bf16_min="$(tensorcash_bf16_single_min_vram_mib)"
  fp8_min="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2_min="$(tensorcash_fp8_tp2_min_vram_mib)"
  tp2_min="$(tensorcash_bf16_tp2_min_vram_mib)"
  tp4_min="$(tensorcash_bf16_tp4_min_vram_mib)"
  precision_mode="$(tensorcash_precision_mode)" || fail "Invalid TensorCash precision configuration."
  mapfile -t memories < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
  ((${#memories[@]} > 0)) || fail "No NVIDIA GPUs are visible on this host."

  for index in "${!memories[@]}"; do
    memory="${memories[$index]}"
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] || fail "Could not read memory for GPU $index."
    detected+=("$index=${memory}MiB")
    case "$precision_mode" in
      auto)
        if (( memory >= fp8_min )); then
          singles+=("$index")
        elif (( memory >= tp2_min )); then
          tp2+=("$index")
        elif (( memory >= fp8_tp2_min )); then
          fp8_tp2+=("$index")
        elif (( memory >= tp4_min )); then
          tp4+=("$index")
        else
          leftovers+=("$index")
        fi
        ;;
      bf16)
        if (( memory >= bf16_min )); then
          singles+=("$index")
        elif (( memory >= tp2_min )); then
          tp2+=("$index")
        elif (( memory >= tp4_min )); then
          tp4+=("$index")
        else
          leftovers+=("$index")
        fi
        ;;
      fp8)
        if (( memory >= fp8_min )); then
          singles+=("$index")
        elif (( memory >= fp8_tp2_min )); then
          fp8_tp2+=("$index")
        else
          leftovers+=("$index")
        fi
        ;;
    esac
  done

  echo "TensorCash detected GPU VRAM: ${detected[*]}" >&2

  for index in "${singles[@]}"; do
    groups+=("$index")
  done
  for ((start = 0; start + 1 < ${#tp2[@]}; start += 2)); do
    groups+=("${tp2[$start]},${tp2[$((start + 1))]}")
  done
  for ((start = 0; start + 1 < ${#fp8_tp2[@]}; start += 2)); do
    groups+=("${fp8_tp2[$start]},${fp8_tp2[$((start + 1))]}")
  done
  for ((start = 0; start + 3 < ${#tp4[@]}; start += 4)); do
    groups+=("${tp4[$start]},${tp4[$((start + 1))]},${tp4[$((start + 2))]},${tp4[$((start + 3))]}")
  done
  for ((start = (${#tp2[@]} / 2) * 2; start < ${#tp2[@]}; start += 1)); do
    leftovers+=("${tp2[$start]}")
  done
  for ((start = (${#fp8_tp2[@]} / 2) * 2; start < ${#fp8_tp2[@]}; start += 1)); do
    leftovers+=("${fp8_tp2[$start]}")
  done
  for ((start = (${#tp4[@]} / 4) * 4; start < ${#tp4[@]}; start += 1)); do
    leftovers+=("${tp4[$start]}")
  done

  ((${#groups[@]} > 0)) || fail "No valid TensorCash group: auto mode uses one >=22 GiB BF16 GPU, one >=11.5 GiB FP8 GPU, two >=6 GiB FP8 GPUs, two >=11 GiB BF16 GPUs, or four >=7.5 GiB BF16 GPUs."
  if ((${#leftovers[@]} > 0)); then
    echo "Auto planner leaves GPU(s) ${leftovers[*]} idle because TensorCash requires TP=1, 2, or 4 groups." >&2
  fi
  local IFS=';'
  printf '%s\n' "${groups[*]}"
}

resolve_group_runtime_profile() {
  local group="$1" gpu memory min_memory=0
  local -a group_gpus=()
  IFS=',' read -r -a group_gpus <<< "$group"
  for gpu in "${group_gpus[@]}"; do
    memory="$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] || fail "Could not read VRAM for GPU $gpu."
    (( min_memory == 0 || memory < min_memory )) && min_memory="$memory"
  done
  GROUP_MIN_MEMORY="$min_memory"
  GROUP_MODEL_PRECISION="$(tensorcash_resolve_precision "$min_memory" "${#group_gpus[@]}")" || \
    fail "GPU group $group cannot satisfy the selected TensorCash precision profile."
  GROUP_VLLM_QUANTIZATION="$(tensorcash_vllm_quantization "$GROUP_MODEL_PRECISION")"
}

configure_auto_group_concurrency() {
  local group="$1" start cap prefetch prefetch_raw required_buffer fp8_start fp8_step
  local -a group_gpus=()
  IFS=',' read -r -a group_gpus <<< "$group"
  cap="${TENSORCASH_AUTO_CONCURRENCY_CEILING:-1024}"
  [[ "$cap" =~ ^[1-9][0-9]*$ ]] && (( cap <= 1024 )) || \
    fail "TENSORCASH_AUTO_CONCURRENCY_CEILING must be an integer between 1 and 1024."
  start="${TENSORCASH_AUTO_CONCURRENCY_START:-32}"
  [[ "$start" =~ ^[1-9][0-9]*$ ]] || fail "TENSORCASH_AUTO_CONCURRENCY_START must be a positive integer."
  (( start <= cap )) || start="$cap"
  # Two low-VRAM FP8 ranks can boot far below the normal 32-request seed.
  # Start from a conservative verified floor and climb in small steps; vLLM's
  # own capacity bootstrap remains the final arbiter for a particular driver.
  AUTO_SIDECAR_STEP="${TENSORCASH_AUTO_CONCURRENCY_STEP}"
  if [[ "$GROUP_MODEL_PRECISION" == fp8 && ${#group_gpus[@]} -eq 2 ]]; then
    if (( GROUP_MIN_MEMORY < 7000 )); then
      fp8_start=8
      fp8_step=8
    else
      fp8_start=16
      fp8_step=16
    fi
    (( start > fp8_start )) && start="$fp8_start"
    (( AUTO_SIDECAR_STEP > fp8_step )) && AUTO_SIDECAR_STEP="$fp8_step"
  fi
  # A queued vLLM reserve keeps fixed-length completion cohorts from draining
  # the GPU between sidecar refill callbacks. It uses no extra running KV
  # slots; explicit numeric settings remain an operator override.
  prefetch_raw="${NOMP_SIDECAR_PREFETCH_REQUESTS:-auto}"
  if [[ "$prefetch_raw" == "auto" ]]; then
    prefetch="$(( (cap + 3) / 4 ))"
    (( prefetch <= 256 )) || prefetch=256
  else
    prefetch="$prefetch_raw"
    [[ "$prefetch" =~ ^[0-9]+$ ]] || fail "NOMP_SIDECAR_PREFETCH_REQUESTS must be auto or a non-negative integer."
  fi

  AUTO_VLLM_MAX_NUM_SEQS="$cap"
  # TensorCash's sampler owns a row-indexed proof ring. vLLM may sample both
  # running rows and its local waiting reserve in one pass, so the ring must
  # cover their sum rather than the historical fixed 1024-row default.
  AUTO_POW_MAX_CONCURRENCY="$(( cap + prefetch ))"
  AUTO_VLLM_CUDA_GRAPH_SIZES="$(vllm_cuda_graph_sizes "$cap")"
  AUTO_VLLM_MAX_NUM_BATCHED_TOKENS=""
  # Match the validated single-24-GiB profile. TP groups made from smaller
  # cards retain vLLM's default budget because their activation headroom is
  # topology-dependent and must not be guessed by the launcher.
  [[ ${#group_gpus[@]} -eq 1 ]] && AUTO_VLLM_MAX_NUM_BATCHED_TOKENS=8192
  AUTO_SIDECAR_START="$start"
  AUTO_SIDECAR_PREFETCH="$prefetch"
  AUTO_SIDECAR_MIN_BUFFERED="$(( start > 4 ? start / 2 : 2 ))"
  AUTO_SIDECAR_MAX_BUFFERED="$(( cap * 2 ))"
  (( AUTO_SIDECAR_MAX_BUFFERED <= 512 )) || AUTO_SIDECAR_MAX_BUFFERED=512
  required_buffer="$(( cap + prefetch ))"
  (( required_buffer <= 512 )) || required_buffer=512
  (( AUTO_SIDECAR_MAX_BUFFERED >= required_buffer )) || \
    fail "Auto proof buffer cannot cover the configured NOMP_SIDECAR_PREFETCH_REQUESTS."
}

vllm_cuda_graph_sizes() {
  local max_seqs="$1" size
  local -a sizes=(1 2 4 8 16 32 64 96 128 160 192 224 256 320 384 448 512 640 768 896 1024)
  local -a applicable=()
  for size in "${sizes[@]}"; do
    (( size <= max_seqs )) && applicable+=("$size")
  done
  (IFS=,; printf '%s' "${applicable[*]}")
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
  if [[ "${GPU_GROUPS:-}" == auto ]]; then
    require_command nvidia-smi
    GPU_GROUPS="$(auto_gpu_groups)"
  fi
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
  gpu_groups="${groups_arg:-auto}"
  [[ "$gpu_groups" == auto || "$gpu_groups" =~ ^[0-9]+(,[0-9]+)*(;[0-9]+(,[0-9]+)*)*$ ]] || fail "--gpu-groups must be auto or look like 0,1,2,3 or 0,1;2,3."
  token="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  umask 077
  cat > "$config" <<EOF
MINER_IMAGE=$(default_runtime_image)
POOL_HOST=${pool_arg%:*}
POOL_PORT=${pool_arg##*:}
PAYOUT_ACCOUNT=$wallet_arg
WORKER=$worker_arg
NOMP_SIDECAR_TOKEN=$token
MODEL_NAME=Qwen/Qwen3-8B
MODEL_COMMIT=9c925d64d72725edaf899c6cb9c377fd0709d9c5
MODEL_DIFFICULTY_NORMALIZER=1000000
MAX_MODEL_LEN=2048
# Every mining start force-syncs this launcher to origin/main and re-execs the
# updated script. Set false only for an emergency offline recovery.
TENSORCASH_AUTO_UPDATE=true
# auto = FP8 on 12/16 GiB TP=1 cards and BF16 on >=22 GiB TP=1 cards.
# Set fp8 or bf16 only to force a deliberate profile across every group.
TENSORCASH_MODEL_PRECISION=auto
# Use the common high-utilization profile for every supported TP group. The
# vLLM bootstrap probe remains the final authority and falls back safely when
# a host cannot admit its requested sequence capacity.
GPU_MEM_UTIL=0.89
# The launcher starts at 32 and lets vLLM admission plus measured throughput
# choose the useful level. Set mode=manual only for a deliberate benchmark.
TENSORCASH_CONCURRENCY_MODE=auto
TENSORCASH_AUTO_CONCURRENCY_START=32
TENSORCASH_AUTO_CONCURRENCY_STEP=32
TENSORCASH_AUTO_CONCURRENCY_CEILING=1024
GPU_GROUPS=$gpu_groups
MODELS_DATA=$script_dir/runtime/models
RUNTIME_DATA=$script_dir/runtime/data
TENSORCASH_POLL_MS=200
TENSORCASH_SUBMIT_WINDOW=16
TENSORCASH_STATS_INTERVAL=30
TENSORCASH_SIDECAR_WAIT_SECONDS=1200
TENSORCASH_IMAGE_PULL_ATTEMPTS=12
TENSORCASH_IMAGE_PULL_DELAY_SECONDS=15
TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS=12
TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS=15
# Optional HTTP(S) proxy for the model/archive downloader. Docker image pulls
# need a Docker daemon proxy; run: bash docker-proxy.sh --proxy URL
# TENSORCASH_HTTP_PROXY=http://127.0.0.1:7890
# Optional resumable .tar.zst image archive and its SHA-256 digest.
# TENSORCASH_IMAGE_ARCHIVE_URL=https://mirror.example/tensorcash-image.tar.zst
# TENSORCASH_IMAGE_ARCHIVE_SHA256=replace_with_64_hex_characters
EOF
elif [[ -n "$pool_arg$wallet_arg$worker_arg$groups_arg" ]]; then
  fail "miner.env already exists; edit it explicitly or remove it before changing launch parameters."
fi

set -a
# shellcheck disable=SC1090
source "$config"
set +a

# The legacy v0.10 image has PyTorch kernels through sm_90 only, so it cannot
# run a 5090/Blackwell GPU at all.  This exact known-default migration is safe
# and intentionally does not touch custom image tags or immutable digests.
if has_blackwell_gpu; then
  case "${MINER_IMAGE:-}" in
    "$LEGACY_RUNTIME_IMAGE")
      require_published_blackwell_runtime
      echo "Blackwell GPU detected; replacing incompatible $LEGACY_RUNTIME_IMAGE with $BLACKWELL_RUNTIME_IMAGE"
      sed -i "s|^MINER_IMAGE=.*|MINER_IMAGE=$BLACKWELL_RUNTIME_IMAGE|" "$config"
      MINER_IMAGE="$BLACKWELL_RUNTIME_IMAGE"
      export MINER_IMAGE
      ;;
    "$BLACKWELL_RUNTIME_IMAGE")
      require_published_blackwell_runtime
      ;;
  esac
fi

# Existing miner.env files gain the safe adaptive mode by default. Operators
# can preserve a benchmarked fixed setting with TENSORCASH_CONCURRENCY_MODE=manual.
TENSORCASH_SUBMIT_WINDOW="${TENSORCASH_SUBMIT_WINDOW:-16}"
positive_integer "$TENSORCASH_SUBMIT_WINDOW" TENSORCASH_SUBMIT_WINDOW
(( TENSORCASH_SUBMIT_WINDOW <= 64 )) || \
  fail "TENSORCASH_SUBMIT_WINDOW must not exceed 64."
TENSORCASH_CONCURRENCY_MODE="${TENSORCASH_CONCURRENCY_MODE:-auto}"
TENSORCASH_MODEL_PRECISION="$(tensorcash_precision_mode)" || fail "Invalid TensorCash precision configuration."
tensorcash_validate_vram_thresholds || fail "Invalid TensorCash VRAM threshold configuration."
tensorcash_validate_gpu_mem_util "$TENSORCASH_MODEL_PRECISION" "${GPU_MEM_UTIL:-}" || \
  fail "Invalid GPU_MEM_UTIL configuration."
TENSORCASH_AUTO_CONCURRENCY_START="${TENSORCASH_AUTO_CONCURRENCY_START:-32}"
TENSORCASH_AUTO_CONCURRENCY_STEP="${TENSORCASH_AUTO_CONCURRENCY_STEP:-32}"
TENSORCASH_AUTO_CONCURRENCY_CEILING="${TENSORCASH_AUTO_CONCURRENCY_CEILING:-1024}"
case "$TENSORCASH_CONCURRENCY_MODE" in
  auto)
    [[ "$TENSORCASH_AUTO_CONCURRENCY_START" =~ ^[1-9][0-9]*$ ]] || \
      fail "TENSORCASH_AUTO_CONCURRENCY_START must be a positive integer."
    [[ "$TENSORCASH_AUTO_CONCURRENCY_STEP" =~ ^[1-9][0-9]*$ ]] || \
      fail "TENSORCASH_AUTO_CONCURRENCY_STEP must be a positive integer."
    (( TENSORCASH_AUTO_CONCURRENCY_STEP <= 256 )) || \
      fail "TENSORCASH_AUTO_CONCURRENCY_STEP must not exceed 256."
    positive_integer "$TENSORCASH_AUTO_CONCURRENCY_CEILING" TENSORCASH_AUTO_CONCURRENCY_CEILING
    (( TENSORCASH_AUTO_CONCURRENCY_CEILING <= 1024 )) || \
      fail "TENSORCASH_AUTO_CONCURRENCY_CEILING must not exceed 1024."
    ;;
  manual)
    positive_integer "${VLLM_MAX_NUM_SEQS:-}" VLLM_MAX_NUM_SEQS
    positive_integer "${NOMP_SIDECAR_CONCURRENCY:-}" NOMP_SIDECAR_CONCURRENCY
    (( NOMP_SIDECAR_CONCURRENCY <= VLLM_MAX_NUM_SEQS )) || \
      fail "NOMP_SIDECAR_CONCURRENCY must not exceed VLLM_MAX_NUM_SEQS."
    (( NOMP_SIDECAR_CONCURRENCY <= 1024 )) || \
      fail "NOMP_SIDECAR_CONCURRENCY must not exceed 1024."
    ;;
  *) fail "TENSORCASH_CONCURRENCY_MODE must be auto or manual." ;;
esac

if "$update_only"; then
  if has_blackwell_gpu; then
    require_published_blackwell_runtime
    update_image="${MINER_UPDATE_IMAGE:-ghcr.io/avalonbtc/tensorcash-miner:mainnet-blackwell-latest}"
  else
    update_image="${MINER_UPDATE_IMAGE:-ghcr.io/avalonbtc/tensorcash-miner:mainnet-latest}"
  fi
  echo "Checking for a new miner runtime: $update_image"
  pull_image_with_retries "$update_image"
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
if [[ "$GPU_GROUPS" == auto ]]; then
  GPU_GROUPS="$(auto_gpu_groups)"
  echo "Auto-selected TensorCash GPU groups: $GPU_GROUPS"
fi
[[ "$GPU_GROUPS" =~ ^[0-9]+(,[0-9]+)*(;[0-9]+(,[0-9]+)*)*$ ]] || fail "Invalid GPU_GROUPS in miner.env."

IFS=';' read -r -a group_list <<< "$GPU_GROUPS"
declare -A seen_gpu=()
for group in "${group_list[@]}"; do
  IFS=',' read -r -a group_gpus <<< "$group"
  case "${#group_gpus[@]}" in
    1|2|4|8) ;;
    *) fail "TensorCash TP group '$group' has ${#group_gpus[@]} GPUs; use TP=1, 2, 4, or 8 only." ;;
  esac
  for gpu in "${group_gpus[@]}"; do
    [[ "$gpu" -lt "$gpu_count" ]] || fail "GPU $gpu does not exist; host exposes $gpu_count GPU(s)."
    [[ -z "${seen_gpu[$gpu]:-}" ]] || fail "GPU $gpu appears in more than one group."
    seen_gpu[$gpu]=1
  done
done

mkdir -p "$MODELS_DATA" "$RUNTIME_DATA"
# The sidecar launches vLLM as the unprivileged `worker` user.  Model weights
# are public, immutable inputs, so make the host cache traversable/readable
# while keeping runtime proof data private to the host owner.
chmod 755 "$MODELS_DATA"
find "$MODELS_DATA" -type d -exec chmod a+rx {} +
find "$MODELS_DATA" -type f -exec chmod a+r {} +
chmod 700 "$RUNTIME_DATA"
if [[ "${TENSORCASH_SKIP_IMAGE_PULL:-false}" =~ ^(1|true|yes)$ ]]; then
  docker image inspect "$MINER_IMAGE" >/dev/null 2>&1 || fail "TENSORCASH_SKIP_IMAGE_PULL is set, but $MINER_IMAGE is not loaded locally."
  echo "Using the already-loaded TensorCash image; registry pull skipped."
else
  ensure_runtime_image "$MINER_IMAGE"
fi

ensure_compatible_miner_binary

model_cache_name="${MODEL_NAME//\//--}"
model_snapshot="$MODELS_DATA/hub/models--${model_cache_name}/snapshots/${MODEL_COMMIT}"
model_config="$model_snapshot/config.json"
model_complete="$MODELS_DATA/.tensorcash-model-${model_cache_name}-${MODEL_COMMIT}.complete"
if [[ ! -f "$model_complete" ]]; then
  echo "No completed-model marker exists. Verifying/downloading the full pinned snapshot before starting vLLM..."
  download_model_with_retries "$MODEL_NAME" "$MODEL_COMMIT" "$MODELS_DATA"
  [[ -f "$model_config" ]] || fail "Model downloader returned success without config.json in the pinned snapshot."
  compgen -G "$model_snapshot/*.safetensors" >/dev/null || fail "Model downloader returned success without any safetensors weights."
  umask 077
  printf 'model=%s\ncommit=%s\ncompleted_utc=%s\n' "$MODEL_NAME" "$MODEL_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$model_complete"
fi
[[ -f "$model_config" ]] || fail "The completed model marker exists but config.json is missing: $model_config"
compgen -G "$model_snapshot/*.safetensors" >/dev/null || fail "The completed model marker exists but no safetensors weights are present: $model_snapshot"

safe_worker="${WORKER//[^A-Za-z0-9_-]/-}"
for index in "${!group_list[@]}"; do
  group_number=$((index + 1))
  group="${group_list[$index]}"
  IFS=',' read -r -a group_gpus <<< "$group"
  group_worker="${WORKER}-g${group_number}"
  group_runtime="$RUNTIME_DATA/group-${group_number}"
  mkdir -p "$group_runtime"
  # The image runs vLLM as its unprivileged `worker` user and writes PoW
  # diagnostics under /data/miner_logs.  This per-group directory is the only
  # writable bind mount; sticky mode keeps files isolated between local users.
  chmod 1777 "$group_runtime"
  echo "Starting ${group_worker}: GPUs ${group}, TP=${#group_gpus[@]}"
  (
    export WORKER="$group_worker"
    export NVIDIA_VISIBLE_DEVICES="$group"
    export VLLM_TENSOR_PARALLEL_SIZE="${#group_gpus[@]}"
    export VLLM_MODEL_PATH="/models/hub/models--${model_cache_name}/snapshots/${MODEL_COMMIT}"
    export RUNTIME_DATA="$group_runtime"
    resolve_group_runtime_profile "$group"
    tensorcash_validate_gpu_mem_util "$GROUP_MODEL_PRECISION" "$GPU_MEM_UTIL" || \
      fail "GPU_MEM_UTIL is not valid for ${GROUP_MODEL_PRECISION} on GPU group $group."
    export TENSORCASH_MODEL_PRECISION="$GROUP_MODEL_PRECISION"
    export TENSORCASH_VLLM_QUANTIZATION="$GROUP_VLLM_QUANTIZATION"
    echo "Runtime precision for ${group_worker}: ${GROUP_MODEL_PRECISION} (minimum group VRAM ${GROUP_MIN_MEMORY} MiB)"
    if [[ "$TENSORCASH_CONCURRENCY_MODE" == auto ]]; then
      configure_auto_group_concurrency "$group"
      export VLLM_MAX_NUM_SEQS="$AUTO_VLLM_MAX_NUM_SEQS"
      export POW_MAX_CONCURRENCY="$AUTO_POW_MAX_CONCURRENCY"
      export VLLM_CUDA_GRAPH_SIZES="$AUTO_VLLM_CUDA_GRAPH_SIZES"
      export VLLM_MAX_NUM_BATCHED_TOKENS="$AUTO_VLLM_MAX_NUM_BATCHED_TOKENS"
      export NOMP_SIDECAR_CONCURRENCY=auto
      export NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY="$AUTO_SIDECAR_START"
      export NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY="$AUTO_VLLM_MAX_NUM_SEQS"
      export NOMP_SIDECAR_ADAPTIVE_STEP="$AUTO_SIDECAR_STEP"
      export TENSORCASH_VLLM_FALLBACK_MIN_SEQS="$AUTO_SIDECAR_START"
      export NOMP_SIDECAR_MIN_BUFFERED_PROOFS="$AUTO_SIDECAR_MIN_BUFFERED"
      export NOMP_SIDECAR_MAX_BUFFERED_PROOFS="$AUTO_SIDECAR_MAX_BUFFERED"
      export NOMP_SIDECAR_PREFETCH_REQUESTS="$AUTO_SIDECAR_PREFETCH"
      echo "Auto concurrency for ${group_worker}: start=${AUTO_SIDECAR_START}, cap=${AUTO_VLLM_MAX_NUM_SEQS}, step=${AUTO_SIDECAR_STEP}, prefetch=${AUTO_SIDECAR_PREFETCH}"
    fi
    docker compose --project-name "tensorcash-${safe_worker}-g${group_number}" --env-file "$config" -f "$script_dir/docker-compose.yml" up -d --remove-orphans
  )
done

echo "TensorCash started. Model cache: $MODELS_DATA"
echo "Use: docker ps --filter 'name=tensorcash-'"
