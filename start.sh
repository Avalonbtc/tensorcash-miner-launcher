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
pool_tls_arg=""
pool_tls_insecure_arg=""
stop_only=false
update_only=false
refresh_env_only=false

readonly LEGACY_RUNTIME_IMAGE='ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0'
readonly BLACKWELL_RUNTIME_IMAGE='ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.1-blackwell'
# Keep the launcher's internal schema constant distinct from the public
# MINER_ENV_POLICY_SCHEMA key written to miner.env. The latter is sourced as
# a normal environment variable during every launch and therefore must never
# collide with a readonly shell variable.
readonly TENSORCASH_MINER_ENV_POLICY_SCHEMA=2

usage() {
  cat <<'EOF'
Usage:
  bash start.sh --pool HOST:PORT --wallet PAYOUT --worker NAME [--tls] [--gpu-groups auto|GROUPS]
  bash start.sh --refresh-env
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

initial_runtime_image() {
  local image="${TENSORCASH_INITIAL_MINER_IMAGE:-}"
  if [[ -z "$image" ]]; then
    default_runtime_image
    return 0
  fi
  [[ "$image" =~ ^ghcr\.io/avalonbtc/tensorcash-miner:[A-Za-z0-9._-]+$ ]] || \
    fail "TENSORCASH_INITIAL_MINER_IMAGE must be a tagged Avalonbtc TensorCash runtime."
  printf '%s\n' "$image"
}

blackwell_runtime_published() {
  command -v docker >/dev/null 2>&1 || return 1
  docker manifest inspect "$BLACKWELL_RUNTIME_IMAGE" >/dev/null 2>&1
}

require_published_blackwell_runtime() {
  blackwell_runtime_published && return 0
  fail "RTX 50-series GPU detected, but $BLACKWELL_RUNTIME_IMAGE is not published in GHCR yet. Docker mining is unavailable for this GPU until the Blackwell image build succeeds; use native-vast.sh instead."
}

blackwell_runtime_available() {
  docker image inspect "$BLACKWELL_RUNTIME_IMAGE" >/dev/null 2>&1 || \
    blackwell_runtime_published
}

require_available_blackwell_runtime() {
  blackwell_runtime_available && return 0
  fail "RTX 50-series GPU detected, but $BLACKWELL_RUNTIME_IMAGE is neither loaded locally nor published in GHCR. Use a verified seed bundle or native-vast.sh until the image build succeeds."
}

positive_integer() {
  local value="$1" name="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer."
}

normalize_pool_tls_settings() {
  local setting value
  for setting in POOL_TLS POOL_TLS_INSECURE; do
    value="${!setting:-false}"
    case "${value,,}" in
      1|true|yes) printf -v "$setting" '%s' true ;;
      0|false|no) printf -v "$setting" '%s' false ;;
      *) fail "$setting must be true or false." ;;
    esac
  done
  [[ "$POOL_TLS_INSECURE" != true || "$POOL_TLS" == true ]] || \
    fail "POOL_TLS_INSECURE requires POOL_TLS=true."
}

miner_env_value() {
  local config_path="$1" key="$2"
  sed -n -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//p" "$config_path" | tail -n 1
}

miner_env_needs_policy_refresh() {
  local config_path="$1"
  [[ "$(miner_env_value "$config_path" MINER_ENV_POLICY_SCHEMA)" != "$TENSORCASH_MINER_ENV_POLICY_SCHEMA" ]]
}

refresh_miner_env_policy() {
  local config_path="$1" requested_by="$2" backup_path temp_path timestamp
  local policy_keys policy_pattern

  [[ -f "$config_path" ]] || fail "Cannot refresh a missing miner.env: $config_path"
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  backup_path="${config_path}.before-policy-refresh.${timestamp}"
  temp_path="${config_path}.refresh.$$"
  cp -p "$config_path" "$backup_path"

  # These keys describe the launcher's evolving hardware policy, not miner
  # identity. Keeping old values here can silently force BF16/TP groups on
  # an SM80+ 12--16 GiB card even after the launcher itself has been updated.
  # Keep every other line verbatim: wallet, pool, worker, TLS, token, image,
  # caches, proxies, archive mirrors, and operator-specific settings survive.
  policy_keys='MINER_ENV_POLICY_SCHEMA|GPU_GROUPS|TENSORCASH_MODEL_PRECISION|GPU_MEM_UTIL|TENSORCASH_CONCURRENCY_MODE|TENSORCASH_AUTO_CONCURRENCY_START|TENSORCASH_AUTO_CONCURRENCY_STEP|TENSORCASH_AUTO_CONCURRENCY_CEILING|NOMP_SIDECAR_CONCURRENCY|NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY|NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY|NOMP_SIDECAR_ADAPTIVE_STEP|NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS|NOMP_SIDECAR_PREFETCH_REQUESTS|NOMP_SIDECAR_MIN_BUFFERED_PROOFS|NOMP_SIDECAR_MAX_BUFFERED_PROOFS|VLLM_MAX_NUM_SEQS|VLLM_MAX_NUM_BATCHED_TOKENS|VLLM_CUDA_GRAPH_SIZES|VLLM_TENSOR_PARALLEL_SIZE|VLLM_MODEL_PATH|POW_MAX_CONCURRENCY|TENSORCASH_VLLM_QUANTIZATION|TENSORCASH_VLLM_FALLBACK_MIN_SEQS|TENSORCASH_STATIC_FP8_SNAPSHOT|NVIDIA_VISIBLE_DEVICES'
  policy_pattern="^[[:space:]]*(${policy_keys})[[:space:]]*="
  awk -v pattern="$policy_pattern" '$0 ~ pattern { next } { print }' "$config_path" > "$temp_path"

  cat >> "$temp_path" <<EOF

# TensorCash launcher policy schema. This block is owned by the launcher so
# GPU generations with different precision support do not retain stale rules.
MINER_ENV_POLICY_SCHEMA=$TENSORCASH_MINER_ENV_POLICY_SCHEMA
# auto selects FP8 on supported 6--21.9 GiB SM80+ profiles, FP16 for pre-SM80
# cards, and BF16 for SM80+ cards with >=22 GiB. GPU groups are derived fresh.
TENSORCASH_MODEL_PRECISION=auto
GPU_MEM_UTIL=0.89
TENSORCASH_CONCURRENCY_MODE=auto
TENSORCASH_AUTO_CONCURRENCY_START=32
TENSORCASH_AUTO_CONCURRENCY_STEP=32
TENSORCASH_AUTO_CONCURRENCY_CEILING=1024
NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS=2
GPU_GROUPS=auto
EOF
  chmod 600 "$temp_path"
  mv "$temp_path" "$config_path"
  echo "Refreshed TensorCash GPU policy in $config_path (${requested_by}); identity and connection settings were preserved." >&2
  echo "Previous file saved as: $backup_path" >&2
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
      -e "MODEL_NAME=$model_name" -e "MODEL_COMMIT=$model_commit" \
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

static_fp8_candidate_exists() {
  local index memory capability
  local -a memories=() capabilities=()
  mapfile -t memories < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  mapfile -t capabilities < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || true)
  for index in "${!memories[@]}"; do
    memory="${memories[$index]}"
    memory="${memory//[[:space:]]/}"
    capability="${capabilities[$index]:-}"
    capability="${capability//[[:space:]]/}"
    tensorcash_compute_capability_is_pre_fp8 "$capability" && continue
    tensorcash_static_fp8_tp1_download_needed "$memory" && return 0
  done
  return 1
}

configure_static_fp8_snapshot() {
  local configured snapshot models_root relative
  TENSORCASH_STATIC_FP8_TP1_AVAILABLE=false
  TENSORCASH_STATIC_FP8_CONTAINER_PATH=""
  configured="${TENSORCASH_STATIC_FP8_SNAPSHOT:-}"
  [[ -n "$configured" ]] || return 0

  [[ -d "$configured" && -f "$configured/config.json" && \
     -f "$configured/.tensorcash-static-fp8.complete" ]] || \
    fail "TENSORCASH_STATIC_FP8_SNAPSHOT is incomplete: $configured"
  grep -Fqx "format=tensorcash-static-fp8-v1" "$configured/.tensorcash-static-fp8.complete" && \
  grep -Fqx "model=$MODEL_NAME" "$configured/.tensorcash-static-fp8.complete" && \
  grep -Fqx "commit=$MODEL_COMMIT" "$configured/.tensorcash-static-fp8.complete" && \
  grep -Fqx "artifact_repository=Qwen/Qwen3-8B-FP8" "$configured/.tensorcash-static-fp8.complete" && \
  grep -Fqx "artifact_commit=220b46e3b2180893580a4454f21f22d3ebb187d3" "$configured/.tensorcash-static-fp8.complete" || \
    fail "TENSORCASH_STATIC_FP8_SNAPSHOT does not match the pinned TensorCash model."
  grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"fp8"' "$configured/config.json" || \
    fail "TENSORCASH_STATIC_FP8_SNAPSHOT lacks an FP8 quantization config."
  compgen -G "$configured/*.safetensors" >/dev/null || \
    fail "TENSORCASH_STATIC_FP8_SNAPSHOT has no safetensors weights."

  models_root="$(cd "$MODELS_DATA" && pwd -P)"
  snapshot="$(cd "$configured" && pwd -P)"
  case "$snapshot" in
    "$models_root"/*) relative="${snapshot#"$models_root"/}" ;;
    *) fail "TENSORCASH_STATIC_FP8_SNAPSHOT must be stored below MODELS_DATA so Docker can mount it read-only." ;;
  esac
  TENSORCASH_STATIC_FP8_TP1_AVAILABLE=true
  TENSORCASH_STATIC_FP8_CONTAINER_PATH="/models/$relative"
  export TENSORCASH_STATIC_FP8_TP1_AVAILABLE
  echo "Validated official serialized FP8 snapshot for 12--21.9 GiB TP=1: $snapshot"
}

ensure_static_fp8_snapshot() {
  local repository commit cache_name snapshot marker config
  repository="${TENSORCASH_STATIC_FP8_REPOSITORY:-Qwen/Qwen3-8B-FP8}"
  commit="${TENSORCASH_STATIC_FP8_COMMIT:-220b46e3b2180893580a4454f21f22d3ebb187d3}"
  [[ "$repository" == Qwen/Qwen3-8B-FP8 && "$commit" == 220b46e3b2180893580a4454f21f22d3ebb187d3 ]] || \
    fail "The serialized FP8 TP=1 profile requires the tested Qwen/Qwen3-8B-FP8@220b46e3b2180893580a4454f21f22d3ebb187d3 artifact."
  cache_name="${repository//\//--}"
  snapshot="$MODELS_DATA/hub/models--${cache_name}/snapshots/${commit}"
  # Keep the attestation with the immutable snapshot so a manually seeded
  # cache and a freshly downloaded cache follow exactly the same validation
  # path.  The sidecar never consumes this file; it is launcher metadata.
  marker="$snapshot/.tensorcash-static-fp8.complete"
  config="$snapshot/config.json"
  if [[ ! -f "$marker" ]]; then
    echo "Downloading official serialized FP8 Qwen3-8B for the 12--21.9 GiB TP=1 profile..."
    download_model_with_retries "$repository" "$commit" "$MODELS_DATA"
    [[ -f "$config" ]] || fail "Static FP8 downloader returned without config.json."
    compgen -G "$snapshot/*.safetensors" >/dev/null || \
      fail "Static FP8 downloader returned without safetensors weights."
    grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"fp8"' "$config" || \
      fail "Downloaded Qwen3-8B-FP8 artifact lacks the required FP8 quantization config."
    # The Docker sidecar deliberately runs as an unprivileged user.  The
    # Hugging Face cache was written by the downloader container, so make the
    # immutable model readable before Compose starts the sidecar.
    find "$snapshot" -type d -exec chmod a+rx {} +
    find "$snapshot" -type f -exec chmod a+r {} +
    umask 077
    printf 'format=tensorcash-static-fp8-v1\nmodel=%s\ncommit=%s\nartifact_repository=%s\nartifact_commit=%s\ncompleted_utc=%s\n' \
      "$MODEL_NAME" "$MODEL_COMMIT" "$repository" "$commit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$marker"
  fi
  export TENSORCASH_STATIC_FP8_SNAPSHOT="$snapshot"
  configure_static_fp8_snapshot
}

auto_gpu_groups() {
  local bf16_min fp8_min fp8_tp2_min tp2_min tp4_min
  local -a memories=() capabilities=() singles=() fp8_tp2=() tp2=() legacy_fp16_tp2=() tp4=() groups=() leftovers=() detected=()
  local index memory capability start precision_mode
  tensorcash_validate_vram_thresholds || fail "Invalid TensorCash VRAM threshold configuration."
  bf16_min="$(tensorcash_bf16_single_min_vram_mib)"
  fp8_min="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2_min="$(tensorcash_fp8_tp2_min_vram_mib)"
  tp2_min="$(tensorcash_bf16_tp2_min_vram_mib)"
  tp4_min="$(tensorcash_bf16_tp4_min_vram_mib)"
  precision_mode="$(tensorcash_precision_mode)" || fail "Invalid TensorCash precision configuration."
  mapfile -t memories < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
  mapfile -t capabilities < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' || true)
  ((${#memories[@]} > 0)) || fail "No NVIDIA GPUs are visible on this host."

  for index in "${!memories[@]}"; do
    memory="${memories[$index]}"
    capability="${capabilities[$index]:-unknown}"
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] || fail "Could not read memory for GPU $index."
    detected+=("$index=${memory}MiB/SM${capability}")
    if tensorcash_compute_capability_is_pre_fp8 "$capability"; then
      if [[ "$precision_mode" == fp8 ]]; then
        echo "GPU $index is SM${capability}; FP8 requires SM80 or newer, so it cannot satisfy TENSORCASH_MODEL_PRECISION=fp8." >&2
        leftovers+=("$index")
      elif (( memory >= bf16_min )); then
        singles+=("$index")
      elif (( memory >= tp2_min )); then
        legacy_fp16_tp2+=("$index")
      else
        leftovers+=("$index")
      fi
      continue
    fi
    case "$precision_mode" in
      auto)
        if (( memory >= fp8_min )) || tensorcash_can_use_static_fp8_tp1 "$memory"; then
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
      bf16|fp16)
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
  for ((start = 0; start + 1 < ${#legacy_fp16_tp2[@]}; start += 2)); do
    groups+=("${legacy_fp16_tp2[$start]},${legacy_fp16_tp2[$((start + 1))]}")
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
  for ((start = (${#legacy_fp16_tp2[@]} / 2) * 2; start < ${#legacy_fp16_tp2[@]}; start += 1)); do
    leftovers+=("${legacy_fp16_tp2[$start]}")
  done
  for ((start = (${#fp8_tp2[@]} / 2) * 2; start < ${#fp8_tp2[@]}; start += 1)); do
    leftovers+=("${fp8_tp2[$start]}")
  done
  for ((start = (${#tp4[@]} / 4) * 4; start < ${#tp4[@]}; start += 1)); do
    leftovers+=("${tp4[$start]}")
  done

  ((${#groups[@]} > 0)) || fail "No valid TensorCash group: SM80+ uses one >=22 GiB BF16 GPU, one >=12 GiB serialized-FP8 GPU, two >=6 GiB FP8 GPUs, or BF16 groups; pre-SM80 needs one >=22 GiB FP16 GPU or two >=11 GiB FP16 GPUs."
  if ((${#leftovers[@]} > 0)); then
    echo "Auto planner leaves GPU(s) ${leftovers[*]} idle because TensorCash requires TP=1, 2, or 4 groups." >&2
  fi
  local IFS=';'
  printf '%s\n' "${groups[*]}"
}

resolve_group_runtime_profile() {
  local group="$1" gpu memory capability min_memory=0 legacy_count=0 modern_count=0
  local bf16_min tp2_min
  local -a group_gpus=()
  IFS=',' read -r -a group_gpus <<< "$group"
  for gpu in "${group_gpus[@]}"; do
    memory="$(nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] || fail "Could not read VRAM for GPU $gpu."
    (( min_memory == 0 || memory < min_memory )) && min_memory="$memory"
    capability="$(nvidia-smi --id="$gpu" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
    if tensorcash_compute_capability_is_pre_fp8 "$capability"; then
      legacy_count=$((legacy_count + 1))
    else
      modern_count=$((modern_count + 1))
    fi
  done
  GROUP_MIN_MEMORY="$min_memory"
  if (( legacy_count > 0 )); then
    (( modern_count == 0 )) || fail "GPU group $group mixes pre-SM80 and SM80+ GPUs; do not mix TensorCash precision backends."
    [[ "$TENSORCASH_MODEL_PRECISION" != fp8 ]] || \
      fail "GPU group $group is pre-SM80 and cannot use FP8. Use auto/BF16 with two 12 GiB cards, or an SM80+ GPU."
    bf16_min="$(tensorcash_bf16_single_min_vram_mib)"
    tp2_min="$(tensorcash_bf16_tp2_min_vram_mib)"
    if [[ ${#group_gpus[@]} -eq 1 ]] && (( min_memory >= bf16_min )); then
      GROUP_MODEL_PRECISION=fp16
    elif [[ ${#group_gpus[@]} -eq 2 ]] && (( min_memory >= tp2_min )); then
      GROUP_MODEL_PRECISION=fp16
    else
      fail "GPU group $group is pre-SM80: one GPU needs >=${bf16_min} MiB for FP16, otherwise use exactly two GPUs with >=${tp2_min} MiB each."
    fi
  else
    GROUP_MODEL_PRECISION="$(tensorcash_resolve_precision "$min_memory" "${#group_gpus[@]}")" || \
      fail "GPU group $group cannot satisfy the selected TensorCash precision profile."
  fi
  GROUP_VLLM_QUANTIZATION="$(tensorcash_vllm_quantization "$GROUP_MODEL_PRECISION")"
  GROUP_USES_STATIC_FP8=false
  if [[ ${#group_gpus[@]} -eq 1 && "$GROUP_MODEL_PRECISION" == fp8 ]] && \
      tensorcash_can_use_static_fp8_tp1 "$min_memory"; then
    GROUP_USES_STATIC_FP8=true
  fi
}

prepare_group_runtime_capacity_profile() {
  local runtime_dir="$1" profile_file expected actual
  profile_file="$runtime_dir/vllm-capacity-profile"
  expected="precision=$GROUP_MODEL_PRECISION
tp=$VLLM_TENSOR_PARALLEL_SIZE
min_vram_mib=$GROUP_MIN_MEMORY
max_model_len=$MAX_MODEL_LEN
gpu_mem_util=$GPU_MEM_UTIL
max_num_seqs=$VLLM_MAX_NUM_SEQS
max_batched_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
  actual="$(cat "$profile_file" 2>/dev/null || true)"
  if [[ "$actual" != "$expected" ]]; then
    # A capacity measured against an older context/KV-cache profile is not a
    # safe bootstrap candidate. Clear only the disposable probe result; model
    # and proof data remain untouched.
    rm -f "$runtime_dir/vllm-effective-max-seqs"
    printf '%s\n' "$expected" > "$profile_file"
  fi
}

compose_up_group() {
  local project_name="$1"
  local -a compose_args=(
    --project-name "$project_name"
    --env-file "$config"
    -f "$script_dir/docker-compose.yml"
  )

  # Compose can lose a just-created container while it is resolving a
  # depends_on health condition after an interrupted/old-project teardown.
  # Retry only the affected GPU group from a clean Compose state. This is not
  # an error suppressor: if vLLM/proxy genuinely fails, the second `up` still
  # exits non-zero with its real logs intact.
  if docker compose "${compose_args[@]}" up -d --remove-orphans; then
    return 0
  fi
  echo "Compose startup for $project_name failed; cleaning stale group state and retrying once..." >&2
  docker compose "${compose_args[@]}" down --remove-orphans || true
  docker compose "${compose_args[@]}" up -d --force-recreate --remove-orphans || \
    fail "Compose startup for $project_name failed after a clean group rebuild."
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
  AUTO_MAX_MODEL_LEN="$MAX_MODEL_LEN"
  AUTO_GPU_MEM_UTIL="$GPU_MEM_UTIL"
  AUTO_SIDECAR_STEP="${TENSORCASH_AUTO_CONCURRENCY_STEP}"
  (( start <= cap )) || start="$cap"
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
  # A small queued reserve keeps fixed-length completion cohorts from draining
  # the GPU between sidecar refill callbacks. At very large ceilings, hundreds
  # of waiting jobs become host-side scheduler pressure rather than a useful
  # GPU pipeline, so cap the automatic reserve at 64.
  prefetch_raw="${NOMP_SIDECAR_PREFETCH_REQUESTS:-auto}"
  if [[ "$prefetch_raw" == "auto" ]]; then
    prefetch="$(( (cap + 3) / 4 ))"
    (( prefetch <= 64 )) || prefetch=64
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
  # Single-GPU profiles use a bounded scheduler budget; TP groups retain
  # vLLM's topology-specific admission behavior.
  if [[ ${#group_gpus[@]} -eq 1 ]]; then
    AUTO_VLLM_MAX_NUM_BATCHED_TOKENS=8192
  fi
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
    --tls) pool_tls_arg=true; shift ;;
    --tls-insecure) pool_tls_arg=true; pool_tls_insecure_arg=true; shift ;;
    --gpu-groups) groups_arg="${2:-}"; shift 2 ;;
    --refresh-env) refresh_env_only=true; shift ;;
    --update) update_only=true; shift ;;
    --stop) stop_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

if "$refresh_env_only"; then
  [[ -f "$config" ]] || fail "--refresh-env requires an existing miner.env. Run the first launch with --pool, --wallet, and --worker."
  refresh_miner_env_policy "$config" "explicit --refresh-env"
  exit 0
fi

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
  safe_worker="${WORKER//[^A-Za-z0-9_-]/-}"
  declare -A stop_projects=()
  # Never recompute `auto` groups while stopping. The current model profile can
  # differ from the profile that created the running containers (for example
  # when a static FP8 cache becomes available), so recomputation can omit real
  # g3..gN projects. Discover the labels Docker actually owns instead.
  while IFS= read -r project_name; do
    [[ "$project_name" =~ ^tensorcash-${safe_worker}-g[1-9][0-9]*$ ]] || continue
    stop_projects["$project_name"]=1
  done < <(
    {
      docker ps -a --filter label=com.docker.compose.project \
        --format '{{.Label "com.docker.compose.project"}}'
      docker network ls --filter label=com.docker.compose.project \
        --format '{{.Label "com.docker.compose.project"}}'
    } | sort -u
  )
  if ((${#stop_projects[@]} == 0)); then
    echo "No TensorCash Compose projects found for worker $WORKER."
  else
    for project_name in "${!stop_projects[@]}"; do
      echo "Stopping TensorCash group: $project_name"
      docker compose --project-name "$project_name" --env-file "$config" -f "$script_dir/docker-compose.yml" down --remove-orphans || true
    done
  fi
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
MINER_IMAGE=$(initial_runtime_image)
POOL_HOST=${pool_arg%:*}
POOL_PORT=${pool_arg##*:}
POOL_TLS=${pool_tls_arg:-false}
POOL_TLS_INSECURE=${pool_tls_insecure_arg:-false}
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
# Policy schema owned by the launcher. Existing pre-schema configs are migrated
# once while retaining wallet, pool, worker, token, caches, image and proxies.
MINER_ENV_POLICY_SCHEMA=$TENSORCASH_MINER_ENV_POLICY_SCHEMA
# auto = serialized FP8 on supported SM80+ 12--21.9 GiB TP=1 cards, FP8 TP=2
# on supported 6/8 GiB pairs, FP16 on pre-SM80 profiles, and BF16 on SM80+
# >=22 GiB TP=1 cards. The static FP8 artifact avoids the online BF16-to-FP8
# loading peak that otherwise also breaks 16 GiB cards.
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
elif [[ -n "$pool_arg$wallet_arg$worker_arg$groups_arg$pool_tls_arg$pool_tls_insecure_arg" ]]; then
  fail "miner.env already exists; edit it explicitly or remove it before changing launch parameters."
fi

# Versioned policy ownership makes launcher updates effective without touching
# host identity or deployment settings. Old configs are backed up once before
# their stale grouping/precision overrides are replaced with the current auto
# policy. Operators can repeat this deliberately with --refresh-env.
if miner_env_needs_policy_refresh "$config"; then
  refresh_miner_env_policy "$config" "automatic legacy migration"
fi

set -a
# shellcheck disable=SC1090
source "$config"
set +a

normalize_pool_tls_settings

# The legacy v0.10 image has PyTorch kernels through sm_90 only, so it cannot
# run a 5090/Blackwell GPU at all.  This exact known-default migration is safe
# and intentionally does not touch custom image tags or immutable digests.
if has_blackwell_gpu; then
  case "${MINER_IMAGE:-}" in
    "$LEGACY_RUNTIME_IMAGE")
      require_available_blackwell_runtime
      echo "Blackwell GPU detected; replacing incompatible $LEGACY_RUNTIME_IMAGE with $BLACKWELL_RUNTIME_IMAGE"
      sed -i "s|^MINER_IMAGE=.*|MINER_IMAGE=$BLACKWELL_RUNTIME_IMAGE|" "$config"
      MINER_IMAGE="$BLACKWELL_RUNTIME_IMAGE"
      export MINER_IMAGE
      ;;
    "$BLACKWELL_RUNTIME_IMAGE")
      require_available_blackwell_runtime
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
NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS="${NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS:-2}"
positive_integer "$NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS" NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS
(( NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS <= 4 )) || \
  fail "NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS must not exceed 4."
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
auto_gpu_groups_requested=false
[[ "$GPU_GROUPS" == auto ]] && auto_gpu_groups_requested=true

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
# A 12--21.9 GiB TP=1 FP8 group must use a serialized checkpoint. Do this
# before group planning so these cards avoid the BF16-to-FP8 allocation peak.
if [[ "$TENSORCASH_MODEL_PRECISION" != bf16 ]] && static_fp8_candidate_exists; then
  ensure_static_fp8_snapshot
elif [[ -n "${TENSORCASH_STATIC_FP8_SNAPSHOT:-}" ]]; then
  configure_static_fp8_snapshot
fi

if "$auto_gpu_groups_requested"; then
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

# Do not make an FP8-only rig download the BF16 checkpoint first. A static
# FP8 group has its own verified local path; mixed rigs still fetch the
# canonical checkpoint once for their BF16 groups.
need_canonical_snapshot=false
for group in "${group_list[@]}"; do
  resolve_group_runtime_profile "$group"
  if [[ "$GROUP_USES_STATIC_FP8" != true ]]; then
    need_canonical_snapshot=true
    break
  fi
done

if "$need_canonical_snapshot"; then
  if [[ ! -f "$model_complete" ]]; then
    echo "No completed-model marker exists. Verifying/downloading the full pinned snapshot before starting vLLM..."
    download_model_with_retries "$MODEL_NAME" "$MODEL_COMMIT" "$MODELS_DATA"
    [[ -f "$model_config" ]] || fail "Model downloader returned success without config.json in the pinned snapshot."
    compgen -G "$model_snapshot/*.safetensors" >/dev/null || fail "Model downloader returned success without any safetensors weights."
    find "$model_snapshot" -type d -exec chmod a+rx {} +
    find "$model_snapshot" -type f -exec chmod a+r {} +
    umask 077
    printf 'model=%s\ncommit=%s\ncompleted_utc=%s\n' "$MODEL_NAME" "$MODEL_COMMIT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$model_complete"
  fi
  [[ -f "$model_config" ]] || fail "The completed model marker exists but config.json is missing: $model_config"
  compgen -G "$model_snapshot/*.safetensors" >/dev/null || fail "The completed model marker exists but no safetensors weights are present: $model_snapshot"
else
  echo "All selected GPU groups use the verified serialized FP8 snapshot; skipping the BF16 model download."
fi

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
    export RUNTIME_DATA="$group_runtime"
    resolve_group_runtime_profile "$group"
    if [[ "$GROUP_USES_STATIC_FP8" == true ]]; then
      export VLLM_MODEL_PATH="$TENSORCASH_STATIC_FP8_CONTAINER_PATH"
    else
      "$need_canonical_snapshot" || fail "GPU group $group needs the canonical checkpoint, but it was not prepared."
      export VLLM_MODEL_PATH="/models/hub/models--${model_cache_name}/snapshots/${MODEL_COMMIT}"
    fi
    tensorcash_validate_gpu_mem_util "$GROUP_MODEL_PRECISION" "$GPU_MEM_UTIL" || \
      fail "GPU_MEM_UTIL is not valid for ${GROUP_MODEL_PRECISION} on GPU group $group."
    export TENSORCASH_MODEL_PRECISION="$GROUP_MODEL_PRECISION"
    export TENSORCASH_VLLM_QUANTIZATION="$GROUP_VLLM_QUANTIZATION"
    echo "Runtime precision for ${group_worker}: ${GROUP_MODEL_PRECISION} (minimum group VRAM ${GROUP_MIN_MEMORY} MiB, static_fp8=${GROUP_USES_STATIC_FP8})"
    if [[ "$TENSORCASH_CONCURRENCY_MODE" == auto ]]; then
      configure_auto_group_concurrency "$group"
      export MAX_MODEL_LEN="$AUTO_MAX_MODEL_LEN"
      export GPU_MEM_UTIL="$AUTO_GPU_MEM_UTIL"
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
      echo "Auto concurrency for ${group_worker}: start=${AUTO_SIDECAR_START}, cap=${AUTO_VLLM_MAX_NUM_SEQS}, step=${AUTO_SIDECAR_STEP}, prefetch=${AUTO_SIDECAR_PREFETCH}, context=${MAX_MODEL_LEN}, gpu_mem_util=${GPU_MEM_UTIL}"
    fi
    prepare_group_runtime_capacity_profile "$group_runtime"
    compose_up_group "tensorcash-${safe_worker}-g${group_number}"
  )
done

echo "TensorCash started. Model cache: $MODELS_DATA"
echo "Use: docker ps --filter 'name=tensorcash-'"
