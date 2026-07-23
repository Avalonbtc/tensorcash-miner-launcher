#!/usr/bin/env bash
# TensorCash native launcher for hosted GPU containers without a Docker daemon.
#
# This script intentionally uses the public TensorCash source tree and the
# public, checksum-pinned controller release.  It does not contain or fetch a
# Rust source checkout.  All generated/runtime files live under runtime/native
# and can be deleted with `bash native-vast.sh --purge-runtime`.
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
    exec bash "$script_dir/native-vast.sh" "$@"
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
gpu_arg=""
install_only=false
stop_only=false
logs_only=false
status_only=false
plan_only=false
purge_only=false
rebuild=false

readonly TENSORCASH_SOURCE_URL="${TENSORCASH_SOURCE_URL:-https://github.com/tensorcash/tensorcash.git}"
readonly TENSORCASH_SOURCE_REF="${TENSORCASH_SOURCE_REF:-2df1b0192b8323b1e92699de5f7c0c09c9c5e02a}"
# RTX 50-series needs a CUDA 13 / sm_120 vLLM build.  The legacy v0.10
# Python wheel stops at sm_90 and cannot be repaired with a runtime overlay.
# Keep the source revision explicit so a native Blackwell build is reproducible
# and does not silently follow a moving upstream branch.
readonly TENSORCASH_BLACKWELL_VLLM_URL="${TENSORCASH_BLACKWELL_VLLM_URL:-https://github.com/tensorcash/vllm.git}"
readonly TENSORCASH_BLACKWELL_VLLM_REF="${TENSORCASH_BLACKWELL_VLLM_REF:-a52102827e98fec2f68b2fe7d3d20f08f47452f2}"
readonly TENSORCASH_BLACKWELL_TORCH_VERSION="${TENSORCASH_BLACKWELL_TORCH_VERSION:-2.10.0}"
readonly TENSORCASH_BLACKWELL_TORCHVISION_VERSION="${TENSORCASH_BLACKWELL_TORCHVISION_VERSION:-0.25.0}"
readonly TENSORCASH_BLACKWELL_TORCHAUDIO_VERSION="${TENSORCASH_BLACKWELL_TORCHAUDIO_VERSION:-2.10.0}"
readonly TENSORCASH_BLACKWELL_TORCH_INDEX_URL="${TENSORCASH_BLACKWELL_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

usage() {
  cat <<'EOF'
Usage:
  bash native-vast.sh --pool HOST:PORT --wallet PAYOUT --worker NAME [--gpu INDEX]
  bash native-vast.sh --install
  bash native-vast.sh --plan | --stop | --status | --logs | --purge-runtime

This is for Vast/RunPod-style containers that expose NVIDIA devices but have
no Docker daemon. Native mode runs one TP=1 miner instance for every selected
>=12 GiB GPU, using a serialized FP8 checkpoint on 12--14.9 GiB cards, or a
TP=2 FP8 instance for each pair of 6/8 GiB GPUs. 12--21 GiB cards use FP8
automatically; >=22 GiB cards use BF16. Instances
share one downloaded model/runtime, but use isolated
ports, PIDs, logs, proof data, and worker labels. It is independent from
start.sh's Docker mode.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

positive_integer() {
  local value="$1" name="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer."
}

curl_retry_args() {
  if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    printf '%s\n' '--retry-all-errors'
  fi
}

download_file() {
  local url="$1" output="$2"
  local -a proxy_args=() retry_args=()
  [[ -n "${TENSORCASH_HTTP_PROXY:-}" ]] && proxy_args=(--proxy "$TENSORCASH_HTTP_PROXY")
  mapfile -t retry_args < <(curl_retry_args)
  curl --fail --location --retry 8 "${retry_args[@]}" --connect-timeout 30 \
    --speed-time 90 --speed-limit 10240 "${proxy_args[@]}" --output "$output" "$url"
}

ensure_compatible_miner_binary() {
  local binary_dir="$script_dir/runtime/bin"
  local binary_path="$binary_dir/niuquanminer"
  local binary_url="${TENSORCASH_CONTROLLER_URL:-https://github.com/Avalonbtc/tensorcash-miner-launcher/releases/download/controller-glibc235-v9/niuquanminer-linux-amd64-glibc235}"
  local expected_sha256="${TENSORCASH_CONTROLLER_SHA256:-88fad32fd31782bb9f9dd6ddca516ef7dfb023ecdc349949ac976c3428220c4a}"
  local temporary

  require_command curl
  require_command sha256sum
  [[ "$binary_url" =~ ^https?:// ]] || fail "TENSORCASH_CONTROLLER_URL must be an HTTP(S) URL."
  [[ "$expected_sha256" =~ ^[A-Fa-f0-9]{64}$ ]] || fail "TENSORCASH_CONTROLLER_SHA256 must be a SHA-256 digest."
  mkdir -p "$binary_dir"
  chmod 700 "$binary_dir"
  if [[ -x "$binary_path" ]] && printf '%s  %s\n' "$expected_sha256" "$binary_path" | sha256sum -c - >/dev/null 2>&1; then
    echo "Using verified TensorCash controller: $binary_path"
    return 0
  fi

  temporary="$binary_dir/.niuquanminer.$$"
  rm -f "$temporary"
  echo "Downloading checksum-pinned TensorCash controller (about 3 MB)..."
  download_file "$binary_url" "$temporary" || { rm -f "$temporary"; fail "Could not download TensorCash controller."; }
  printf '%s  %s\n' "$expected_sha256" "$temporary" | sha256sum -c - || { rm -f "$temporary"; fail "Controller checksum mismatch."; }
  chmod 755 "$temporary"
  mv -f "$temporary" "$binary_path"
  echo "Installed verified TensorCash controller."
}

write_initial_config() {
  [[ -n "$pool_arg" && -n "$wallet_arg" && -n "$worker_arg" ]] || {
    usage
    fail "First launch requires --pool, --wallet, and --worker."
  }
  [[ "$pool_arg" =~ ^[A-Za-z0-9.-]+:[1-9][0-9]{0,4}$ ]] || fail "--pool must be HOST:PORT."
  [[ "$wallet_arg" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "--wallet contains unsupported characters."
  [[ "$worker_arg" =~ ^[A-Za-z0-9._-]+$ ]] || fail "--worker contains unsupported characters."
  [[ -z "$gpu_arg" || "$gpu_arg" == all || "$gpu_arg" =~ ^[0-9]+$ ]] || \
    fail "--gpu must be an NVIDIA GPU index or all."

  local token gpu
  token="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
  gpu="${gpu_arg:-all}"
  umask 077
  cat > "$config" <<EOF
# Generated by native-vast.sh. Do not publish this file.
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
# auto = serialized FP8 on 12--14.9 GiB cards, FP8 on 16--21 GiB cards, and
# BF16 on >=22 GiB cards. The 12 GiB path downloads a pinned static checkpoint
# so it does not pay the online BF16-to-FP8 conversion peak.
TENSORCASH_MODEL_PRECISION=auto
# Shared default across native and Docker modes. Startup capacity probing is
# retained as a hard safety gate for every VRAM tier and TP topology.
GPU_MEM_UTIL=0.89
# Starts at 32 and probes upward until real vLLM admission, sustained generation,
# or a request error rejects the next level. Set mode=manual for a fixed benchmark.
TENSORCASH_CONCURRENCY_MODE=auto
TENSORCASH_AUTO_CONCURRENCY_START=32
TENSORCASH_AUTO_CONCURRENCY_STEP=32
# 1024 is an engineering circuit breaker, not a GPU-memory tier cap.
TENSORCASH_AUTO_CONCURRENCY_CEILING=1024
# Require two complete vLLM-rate windows before accepting or rolling back a
# higher concurrency level; this suppresses per-card drift from short samples.
NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS=2
# RTX 50-series selects an isolated CUDA-13/sm_120 runtime automatically.
# Its first --install builds vLLM from source locally; the successful wheel
# is cached under runtime/native/blackwell and reused on later starts.
# TENSORCASH_BLACKWELL_BUILD_JOBS=2
# TENSORCASH_BLACKWELL_AUTO_INSTALL_CUDA_TOOLKIT=true
# Optional scheduler-token override for an intentionally fixed benchmark.
# When unset, native auto mode uses 8192 on 22-39 GiB cards and 65536 on
# >=40 GiB cards, while vLLM still applies its own safe admission limit.
# TENSORCASH_AUTO_MAX_BATCHED_TOKENS=65536
# auto selects >=12 GiB cards singly (with static FP8 on 12--14.9 GiB cards)
# and pairs 6/8 GiB cards for FP8 TP=2.
# A comma-separated list such as 0,2,5 retains legacy independent-card selection.
# --gpu INDEX writes one independent card.
TENSORCASH_NATIVE_GPU_GROUPS=$gpu
# On multi-socket hosts, bind each native group to the NUMA CPU node local to
# its GPU. `auto` silently falls back if the rental container hides topology.
TENSORCASH_NATIVE_NUMA_AFFINITY=auto
# Native NOMP can sustain more than aiohttp's default 100 local connections.
# Raise the soft descriptor limit before starting vLLM/proxy/controller children.
# TENSORCASH_NATIVE_NOFILE_LIMIT=65535
MODELS_DATA=$script_dir/runtime/models
RUNTIME_DATA=$script_dir/runtime/data
TENSORCASH_POLL_MS=200
TENSORCASH_SUBMIT_WINDOW=16
TENSORCASH_STATS_INTERVAL=30
TENSORCASH_SIDECAR_WAIT_SECONDS=1200
TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS=12
TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS=15
EOF
  chmod 600 "$config"
  echo "Created private miner config: $config"
}

load_config() {
  [[ -f "$config" ]] || write_initial_config
  set -a
  # shellcheck disable=SC1090
  source "$config"
  set +a

  # Older native configs predate bounded parallel pool submission. Keep those
  # installations compatible while the controller enforces the same upper cap.
  TENSORCASH_SUBMIT_WINDOW="${TENSORCASH_SUBMIT_WINDOW:-16}"

  [[ "${POOL_HOST:-}" =~ ^[A-Za-z0-9.-]+$ && "${POOL_PORT:-}" =~ ^[1-9][0-9]{0,4}$ ]] || fail "Invalid pool settings in miner.env."
  [[ "${PAYOUT_ACCOUNT:-}" =~ ^[A-Za-z0-9._:-]+$ ]] || fail "Invalid payout account in miner.env."
  [[ "${WORKER:-}" =~ ^[A-Za-z0-9._-]+$ ]] || fail "Invalid worker in miner.env."
  [[ "${NOMP_SIDECAR_TOKEN:-}" =~ ^[A-Fa-f0-9]{32,}$ ]] || fail "Invalid NOMP_SIDECAR_TOKEN in miner.env."
  [[ "${MODEL_NAME:-}" == "Qwen/Qwen3-8B" ]] || fail "Native launcher currently supports the pinned Qwen/Qwen3-8B TensorCash profile only."
  [[ "${MODEL_COMMIT:-}" == "9c925d64d72725edaf899c6cb9c377fd0709d9c5" ]] || fail "MODEL_COMMIT must be the chain-pinned TensorCash Qwen3-8B revision."
  positive_integer "${MODEL_DIFFICULTY_NORMALIZER:-0}" MODEL_DIFFICULTY_NORMALIZER
  positive_integer "${MAX_MODEL_LEN:-0}" MAX_MODEL_LEN
  positive_integer "${TENSORCASH_SUBMIT_WINDOW:-0}" TENSORCASH_SUBMIT_WINDOW
  if [[ -n "${TENSORCASH_AUTO_MAX_BATCHED_TOKENS:-}" ]]; then
    positive_integer "$TENSORCASH_AUTO_MAX_BATCHED_TOKENS" TENSORCASH_AUTO_MAX_BATCHED_TOKENS
    (( TENSORCASH_AUTO_MAX_BATCHED_TOKENS <= 131072 )) || \
      fail "TENSORCASH_AUTO_MAX_BATCHED_TOKENS must not exceed 131072."
  fi
  TENSORCASH_CONCURRENCY_MODE="${TENSORCASH_CONCURRENCY_MODE:-auto}"
  TENSORCASH_AUTO_CONCURRENCY_START="${TENSORCASH_AUTO_CONCURRENCY_START:-32}"
  TENSORCASH_AUTO_CONCURRENCY_STEP="${TENSORCASH_AUTO_CONCURRENCY_STEP:-32}"
  TENSORCASH_AUTO_CONCURRENCY_CEILING="${TENSORCASH_AUTO_CONCURRENCY_CEILING:-1024}"
  NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS="${NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS:-2}"
  positive_integer "$NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS" NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS
  (( NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS <= 4 )) || \
    fail "NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS must not exceed 4."
  case "$TENSORCASH_CONCURRENCY_MODE" in
    auto)
      positive_integer "$TENSORCASH_AUTO_CONCURRENCY_START" TENSORCASH_AUTO_CONCURRENCY_START
      positive_integer "$TENSORCASH_AUTO_CONCURRENCY_STEP" TENSORCASH_AUTO_CONCURRENCY_STEP
      (( TENSORCASH_AUTO_CONCURRENCY_STEP <= 256 )) || \
        fail "TENSORCASH_AUTO_CONCURRENCY_STEP must not exceed 256."
      positive_integer "$TENSORCASH_AUTO_CONCURRENCY_CEILING" TENSORCASH_AUTO_CONCURRENCY_CEILING
      (( TENSORCASH_AUTO_CONCURRENCY_CEILING <= 1024 )) || \
        fail "TENSORCASH_AUTO_CONCURRENCY_CEILING must not exceed 1024."
      ;;
    manual)
      positive_integer "${VLLM_MAX_NUM_SEQS:-0}" VLLM_MAX_NUM_SEQS
      positive_integer "${NOMP_SIDECAR_CONCURRENCY:-0}" NOMP_SIDECAR_CONCURRENCY
      positive_integer "${NOMP_SIDECAR_MIN_BUFFERED_PROOFS:-0}" NOMP_SIDECAR_MIN_BUFFERED_PROOFS
      positive_integer "${NOMP_SIDECAR_MAX_BUFFERED_PROOFS:-0}" NOMP_SIDECAR_MAX_BUFFERED_PROOFS
      (( NOMP_SIDECAR_CONCURRENCY <= VLLM_MAX_NUM_SEQS )) || \
        fail "NOMP_SIDECAR_CONCURRENCY must not exceed VLLM_MAX_NUM_SEQS."
      (( NOMP_SIDECAR_CONCURRENCY <= 1024 )) || \
        fail "Native TensorCash manual concurrency must not exceed 1024."
      ;;
    *) fail "TENSORCASH_CONCURRENCY_MODE must be auto or manual." ;;
  esac
  if [[ -n "${NOMP_SIDECAR_ADMISSION_SPREAD_MS:-}" ]]; then
    [[ "$NOMP_SIDECAR_ADMISSION_SPREAD_MS" =~ ^[0-9]+$ ]] || \
      fail "NOMP_SIDECAR_ADMISSION_SPREAD_MS must be a non-negative integer."
    (( NOMP_SIDECAR_ADMISSION_SPREAD_MS <= 30000 )) || \
      fail "NOMP_SIDECAR_ADMISSION_SPREAD_MS must not exceed 30000."
  fi
  if [[ -n "${NOMP_SIDECAR_PREFETCH_REQUESTS:-}" ]]; then
    if [[ "$NOMP_SIDECAR_PREFETCH_REQUESTS" != auto ]]; then
      [[ "$NOMP_SIDECAR_PREFETCH_REQUESTS" =~ ^[0-9]+$ ]] || \
        fail "NOMP_SIDECAR_PREFETCH_REQUESTS must be auto or a non-negative integer."
      (( NOMP_SIDECAR_PREFETCH_REQUESTS <= 256 )) || \
        fail "NOMP_SIDECAR_PREFETCH_REQUESTS must not exceed 256."
    fi
  fi
  TENSORCASH_MODEL_PRECISION="$(tensorcash_precision_mode)" || fail "Invalid TensorCash precision configuration."
  tensorcash_validate_vram_thresholds || fail "Invalid TensorCash VRAM threshold configuration."
  tensorcash_validate_gpu_mem_util "$TENSORCASH_MODEL_PRECISION" "${GPU_MEM_UTIL:-}" || \
    fail "Invalid GPU_MEM_UTIL configuration."
  if [[ "$TENSORCASH_CONCURRENCY_MODE" == manual ]]; then
    (( NOMP_SIDECAR_MIN_BUFFERED_PROOFS <= NOMP_SIDECAR_MAX_BUFFERED_PROOFS )) || \
      fail "NOMP_SIDECAR_MIN_BUFFERED_PROOFS must not exceed NOMP_SIDECAR_MAX_BUFFERED_PROOFS."
  fi
  (( TENSORCASH_SUBMIT_WINDOW <= 64 )) || fail "TENSORCASH_SUBMIT_WINDOW must not exceed 64."
  TENSORCASH_NATIVE_NUMA_AFFINITY="${TENSORCASH_NATIVE_NUMA_AFFINITY:-auto}"
  case "$TENSORCASH_NATIVE_NUMA_AFFINITY" in
    auto|true|false) ;;
    *) fail "TENSORCASH_NATIVE_NUMA_AFFINITY must be auto, true, or false." ;;
  esac
  # Existing native configs did not contain this key. `auto` now selects every
  # viable single card, then creates matching TP=2 groups for lower VRAM.
  TENSORCASH_NATIVE_GPU_GROUPS="${TENSORCASH_NATIVE_GPU_GROUPS:-auto}"
  [[ "$TENSORCASH_NATIVE_GPU_GROUPS" == auto || "$TENSORCASH_NATIVE_GPU_GROUPS" == all || \
     "$TENSORCASH_NATIVE_GPU_GROUPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    fail "TENSORCASH_NATIVE_GPU_GROUPS must be auto, all, or comma-separated GPU indices."
}

ensure_native_open_file_limit() {
  local requested hard current effective
  requested="${TENSORCASH_NATIVE_NOFILE_LIMIT:-65535}"
  positive_integer "$requested" TENSORCASH_NATIVE_NOFILE_LIMIT
  hard="$(ulimit -Hn)"
  current="$(ulimit -Sn)"
  [[ "$hard" =~ ^[1-9][0-9]*$ && "$current" =~ ^[1-9][0-9]*$ ]] || \
    fail "Could not read the native open-file limit."
  effective="$requested"
  (( effective <= hard )) || effective="$hard"
  (( effective >= 4096 )) || \
    fail "Native TensorCash needs a hard open-file limit of at least 4096; current hard limit is $hard."
  if (( current < effective )); then
    ulimit -Sn "$effective" || \
      fail "Could not raise the native open-file limit to $effective."
  fi
  echo "Native open-file limit: $(ulimit -Sn) (hard=$hard)"
}

gpu_is_blackwell() {
  local index="$1" capability name
  capability="$(nvidia-smi --id="$index" --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "$capability" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    awk -v capability="$capability" 'BEGIN { exit !(capability >= 12.0) }' && return 0
    return 1
  fi
  name="$(nvidia-smi --id="$index" --query-gpu=name --format=csv,noheader 2>/dev/null | tr -d '\r' || true)"
  [[ "$name" =~ (RTX[[:space:]]50[0-9]0|Blackwell|GB10) ]]
}

native_runtime_profile() {
  local index count
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')"
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || fail "No NVIDIA GPUs are visible to nvidia-smi."
  for ((index = 0; index < count; index += 1)); do
    if gpu_is_blackwell "$index"; then
      printf '%s\n' blackwell
      return 0
    fi
  done
  printf '%s\n' legacy
}

blackwell_cuda_home() {
  local candidate version
  for candidate in "${TENSORCASH_BLACKWELL_CUDA_HOME:-}" /usr/local/cuda-13.0 /usr/local/cuda; do
    [[ -n "$candidate" && -x "$candidate/bin/nvcc" ]] || continue
    version="$("$candidate/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | tail -n 1)"
    [[ "$version" =~ ^[0-9]+$ ]] && (( version >= 13 )) || continue
    printf '%s\n' "$candidate"
    return 0
  done
  return 1
}

native_paths() {
  local native_root
  native_root="${TENSORCASH_NATIVE_HOME:-$script_dir/runtime/native}"
  NATIVE_PROFILE="$(native_runtime_profile)"
  # Keep a Blackwell ABI isolated from every existing v0.10 native runtime.
  # It has a different Python/vLLM/CUDA extension set and must never overwrite
  # an Ada/Ampere environment in place.
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    NATIVE_HOME="$native_root/blackwell"
  else
    NATIVE_HOME="$native_root"
  fi
  NATIVE_VENV="$NATIVE_HOME/venv"
  NATIVE_SOURCE="$NATIVE_HOME/tensorcash-source"
  NATIVE_VLLM_SOURCE="$NATIVE_SOURCE/services/miner-api/vllm-v010"
  [[ "$NATIVE_PROFILE" == blackwell ]] && \
    NATIVE_VLLM_SOURCE="$NATIVE_SOURCE/services/miner-api/vllm-v019"
  NATIVE_PROXY="$NATIVE_HOME/miner-proxy/src"
  NATIVE_BUILD="$NATIVE_HOME/build"
  NATIVE_INSTANCES="$NATIVE_HOME/instances"
  NATIVE_MARKER="$NATIVE_HOME/.runtime-ready"
  NATIVE_PY="$NATIVE_VENV/bin/python"
  NATIVE_VLLM="$NATIVE_VENV/bin/vllm"
}

native_instance_paths() {
  local instance="$1"
  NATIVE_INSTANCE="$instance"
  NATIVE_INSTANCE_HOME="$NATIVE_INSTANCES/$instance"
  NATIVE_LOGS="$NATIVE_INSTANCE_HOME/logs"
  NATIVE_PIDS="$NATIVE_INSTANCE_HOME/pids"
  NATIVE_INSTANCE_ENV="$NATIVE_INSTANCE_HOME/runtime.env"
}

native_gpu_cpu_affinity() {
  # GPU inference launches a CPU-heavy engine, proxy and proof controller. On
  # multi-socket hosts Linux otherwise migrates those processes across sockets
  # even when every GPU is attached to one NUMA node. Preserve locality with
  # taskset when sysfs exposes a usable GPU-local CPU list; fall back silently
  # on restricted rental containers rather than making mining unavailable.
  local gpu="$1" bus device_path node cpus
  [[ "$TENSORCASH_NATIVE_NUMA_AFFINITY" != false ]] || return 1
  command -v taskset >/dev/null 2>&1 || return 1
  bus="$(nvidia-smi --id="$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr -d '[:space:]')"
  [[ "$bus" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]+$ ]] || return 1
  bus="${bus,,}"
  device_path="/sys/bus/pci/devices/0000:${bus#*:}"
  [[ -r "$device_path/numa_node" ]] || return 1
  node="$(<"$device_path/numa_node")"
  [[ "$node" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/sys/devices/system/node/node${node}/cpulist" ]] || return 1
  cpus="$(<"/sys/devices/system/node/node${node}/cpulist")"
  [[ "$cpus" =~ ^[0-9,-]+$ ]] || return 1
  taskset -c "$cpus" true >/dev/null 2>&1 || return 1
  printf '%s\n' "$cpus"
}

native_group_cpu_affinity() {
  local group="$1" gpu affinity candidate
  local -a gpus=()
  [[ "$TENSORCASH_NATIVE_NUMA_AFFINITY" != false ]] || return 1
  IFS=',' read -r -a gpus <<< "$group"
  for gpu in "${gpus[@]}"; do
    candidate="$(native_gpu_cpu_affinity "$gpu" || true)"
    [[ -n "$candidate" ]] || return 1
    if [[ -z "${affinity:-}" ]]; then
      affinity="$candidate"
    elif [[ "$affinity" != "$candidate" ]]; then
      # A TP group spanning NUMA nodes has no single local CPU set. Leaving
      # it unbound is preferable to pinning half of its ranks remotely.
      return 1
    fi
  done
  [[ -n "${affinity:-}" ]] && printf '%s\n' "$affinity"
}

native_capacity_file() {
  local gpu_name="$1" memory="$2" ceiling="${3:-${TENSORCASH_AUTO_CONCURRENCY_CEILING:-1024}}" precision="${4:-bf16}" tensor_parallel_size="${5:-1}"
  local max_model_len="${6:-$MAX_MODEL_LEN}" gpu_mem_util="${7:-$GPU_MEM_UTIL}" checkpoint_kind="${8:-canonical}" fingerprint
  # Cards with the same model/VRAM/runtime profile can reuse the first card's
  # measured capacity. A failed reuse falls back to normal vLLM discovery.
  fingerprint="${gpu_name}|${memory}|TP${tensor_parallel_size}|${MODEL_COMMIT}|${gpu_mem_util}|${max_model_len}|${precision}|${checkpoint_kind}|${ceiling}"
  fingerprint="$(printf '%s' "$fingerprint" | sha256sum | awk '{print $1}')"
  printf '%s/capacity-%s.txt\n' "$NATIVE_HOME/capacity" "$fingerprint"
}

native_port_available() {
  local port="$1"
  "$NATIVE_PY" - "$port" <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

native_allocate_instance_ports() {
  local instance_number="$1" offset="$((instance_number - 1))"
  local default_offset="$offset" vllm sidecar collector reserved
  for ((; offset < 2048; offset += 1)); do
    vllm="$((8000 + offset))"
    sidecar="$((8080 + offset))"
    collector="$((7002 + offset))"
    (( collector <= 65535 )) || break
    reserved=" ${NATIVE_RESERVED_PORTS:-} "
    [[ "$reserved" == *" $vllm "* || "$reserved" == *" $sidecar "* || "$reserved" == *" $collector "* ]] && continue
    native_port_available "$vllm" || continue
    native_port_available "$sidecar" || continue
    native_port_available "$collector" || continue
    NATIVE_VLLM_PORT="$vllm"
    NATIVE_SIDECAR_PORT="$sidecar"
    NATIVE_COLLECTOR_PORT="$collector"
    NATIVE_RESERVED_PORTS="${NATIVE_RESERVED_PORTS:-} $vllm $sidecar $collector"
    if (( offset != default_offset )); then
      echo "Native g$instance_number bypasses occupied default ports; using vLLM=$vllm sidecar=$sidecar proof=$collector." >&2
    fi
    return 0
  done
  fail "Could not allocate an isolated local port triplet for native group g$instance_number."
}

native_group_profile() {
  local group="$1" count index memory gpu_name min_memory=0
  local -a group_gpus=()
  IFS=',' read -r -a group_gpus <<< "$group"
  count="${#group_gpus[@]}"
  case "$count" in
    1|2) ;;
    *) fail "Native TensorCash supports TP=1 or TP=2 groups only." ;;
  esac
  NATIVE_GROUP_GPU_NAME=""

  for index in "${group_gpus[@]}"; do
    [[ "$index" =~ ^[0-9]+$ ]] || fail "Invalid GPU index '$index' in native group '$group'."
    if [[ "$NATIVE_PROFILE" == blackwell ]] && ! gpu_is_blackwell "$index"; then
      fail "GPU $index is not a 50-series/Blackwell GPU, but this host selected the isolated Blackwell runtime."
    fi
    memory="$(nvidia-smi --id="$index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
    gpu_name="$(nvidia-smi --id="$index" --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ "$memory" =~ ^[1-9][0-9]*$ && -n "$gpu_name" ]] || fail "GPU $index is not visible to nvidia-smi."
    (( min_memory == 0 || memory < min_memory )) && min_memory="$memory"
    [[ -n "${NATIVE_GROUP_GPU_NAME:-}" ]] || NATIVE_GROUP_GPU_NAME="$gpu_name"
  done

  NATIVE_GROUP_GPU_IDS="$group"
  NATIVE_GROUP_TENSOR_PARALLEL_SIZE="$count"
  NATIVE_GROUP_MIN_MEMORY="$min_memory"
  NATIVE_GROUP_PRECISION="$(tensorcash_resolve_precision "$min_memory" "$count")" || \
    fail "Native GPU group $group cannot satisfy the selected TensorCash precision profile."
  NATIVE_GROUP_QUANTIZATION="$(tensorcash_vllm_quantization "$NATIVE_GROUP_PRECISION")"
  NATIVE_GROUP_USES_STATIC_FP8=false
  if [[ "$count" == 1 && "$NATIVE_GROUP_PRECISION" == fp8 ]] && \
      tensorcash_can_use_static_fp8_tp1 "$min_memory"; then
    NATIVE_GROUP_USES_STATIC_FP8=true
  fi
}

resolve_native_gpu_groups() {
  local requested="$TENSORCASH_NATIVE_GPU_GROUPS" count index memory
  local fp8_min fp8_tp2_min tp2_min
  local -a selected=() singles=() bf16_tp2=() fp8_tp2=() groups=() leftovers=()
  local start
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || fail "No NVIDIA GPUs are visible to nvidia-smi."
  tensorcash_validate_vram_thresholds || fail "Invalid TensorCash VRAM threshold configuration."
  fp8_min="$(tensorcash_fp8_single_min_vram_mib)"
  fp8_tp2_min="$(tensorcash_fp8_tp2_min_vram_mib)"
  tp2_min="$(tensorcash_bf16_tp2_min_vram_mib)"

  if [[ "$requested" == auto || "$requested" == all ]]; then
    for ((index = 0; index < count; index += 1)); do
      if [[ "$NATIVE_PROFILE" == blackwell ]] && ! gpu_is_blackwell "$index"; then
        echo "Native Blackwell runtime leaves GPU $index idle (not compute capability 12.x)." >&2
        continue
      fi
      memory="$(nvidia-smi --id="$index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
      [[ "$memory" =~ ^[1-9][0-9]*$ ]] || fail "GPU $index is not visible to nvidia-smi."
      if (( memory >= fp8_min )) || tensorcash_can_use_static_fp8_tp1 "$memory"; then
        singles+=("$index")
      elif (( memory >= tp2_min )); then
        bf16_tp2+=("$index")
      elif (( memory >= fp8_tp2_min )); then
        fp8_tp2+=("$index")
      else
        leftovers+=("$index")
      fi
    done
    for index in "${singles[@]}"; do
      groups+=("$index")
    done
    for ((start = 0; start + 1 < ${#bf16_tp2[@]}; start += 2)); do
      groups+=("${bf16_tp2[$start]},${bf16_tp2[$((start + 1))]}")
    done
    for ((start = 0; start + 1 < ${#fp8_tp2[@]}; start += 2)); do
      groups+=("${fp8_tp2[$start]},${fp8_tp2[$((start + 1))]}")
    done
    for ((start = (${#bf16_tp2[@]} / 2) * 2; start < ${#bf16_tp2[@]}; start += 1)); do
      leftovers+=("${bf16_tp2[$start]}")
    done
    for ((start = (${#fp8_tp2[@]} / 2) * 2; start < ${#fp8_tp2[@]}; start += 1)); do
      leftovers+=("${fp8_tp2[$start]}")
    done
    if ((${#leftovers[@]})); then
      echo "Native auto planner leaves GPU(s) ${leftovers[*]} idle; 12 GiB cards need the serialized FP8 cache or a matching BF16 TP=2 peer, and 6/8 GiB cards need a matching FP8 TP=2 peer." >&2
    fi
  else
    # Compatibility: an existing comma-separated native setting remains a list
    # of independent TP=1 cards. Multi-card FP8 is selected automatically.
    IFS=',' read -r -a selected <<< "$requested"
    declare -A seen=()
    for index in "${selected[@]}"; do
      [[ "$index" =~ ^[0-9]+$ && index -lt count ]] || fail "GPU $index is not visible to nvidia-smi."
      [[ -z "${seen[$index]:-}" ]] || fail "GPU $index appears more than once in TENSORCASH_NATIVE_GPU_GROUPS."
      seen[$index]=1
      groups+=("$index")
    done
  fi

  ((${#groups[@]} > 0)) || fail "No native TensorCash GPU group is eligible: use one >=15 GiB card, one >=12 GiB card with the serialized FP8 cache, a pair of >=11 GiB cards, or a pair of >=6 GiB cards."
  printf '%s\n' "${groups[@]}"
}

configure_native_memory_profile() {
  NATIVE_GROUP_MAX_MODEL_LEN="$MAX_MODEL_LEN"
  NATIVE_GROUP_GPU_MEM_UTIL="$GPU_MEM_UTIL"
}

configure_native_auto_concurrency() {
  local memory="$1" tensor_parallel_size="$2" cap start step prefetch prefetch_raw required_buffer batched_tokens
  # Do not impose a VRAM-tier concurrency cap: vLLM's actual admission and
  # sustained generation choose the useful level.
  cap="$TENSORCASH_AUTO_CONCURRENCY_CEILING"
  start="$TENSORCASH_AUTO_CONCURRENCY_START"
  step="$TENSORCASH_AUTO_CONCURRENCY_STEP"
  # Capacity is not a performance recommendation. A 48 GiB card can admit
  # hundreds of requests, but entering all of them at once can fill the KV
  # cache and hide a large throughput regression from the adaptive controller.
  # Respect the configured starting point on every VRAM tier and grow only
  # after a complete measured interval.
  (( start <= cap )) || start="$cap"
  if [[ "$NATIVE_GROUP_PRECISION" == fp8 && "$tensor_parallel_size" == 2 ]]; then
    if (( memory < 7000 )); then
      (( start > 8 )) && start=8
      (( step > 8 )) && step=8
    else
      (( start > 16 )) && start=16
      (( step > 16 )) && step=16
    fi
  fi
  # Keep a small vLLM waiting reserve. A 25% reserve is useful at 128 running
  # requests, but at a 960-slot ceiling it creates hundreds of additional
  # local HTTP jobs and repeatedly pushes the engine into a cache-saturated
  # cohort. Waiting requests do not use KV slots, yet still consume host
  # scheduling and proxy capacity.
  prefetch_raw="${NOMP_SIDECAR_PREFETCH_REQUESTS:-auto}"
  if [[ "$prefetch_raw" == "auto" ]]; then
    prefetch="$(( (cap + 3) / 4 ))"
    (( prefetch <= 64 )) || prefetch=64
  else
    prefetch="$prefetch_raw"
    [[ "$prefetch" =~ ^[0-9]+$ ]] || fail "NOMP_SIDECAR_PREFETCH_REQUESTS must be auto or a non-negative integer."
  fi

  VLLM_MAX_NUM_SEQS="$cap"
  NOMP_SIDECAR_CONCURRENCY=auto
  NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY="$start"
  NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY="$cap"
  NOMP_SIDECAR_ADAPTIVE_STEP="$step"
  NOMP_SIDECAR_PREFETCH_REQUESTS="$prefetch"
  VLLM_CUDA_GRAPH_SIZES="$(vllm_cuda_graph_sizes "$cap")"
  # `max-num-seqs` is only a ceiling. vLLM also gates active requests by the
  # batched-token budget. 8192 is the validated 22-39 GiB profile; leaving it
  # in place on 48 GiB cards held their real active request count near 100
  # despite the sidecar safely sustaining much more work. Keep the smaller
  # profile unchanged and use 65536 only where >=40 GiB VRAM has the KV-cache
  # headroom. vLLM still refuses any admission that is unsafe at runtime.
  if [[ -n "${TENSORCASH_AUTO_MAX_BATCHED_TOKENS:-}" ]]; then
    batched_tokens="$TENSORCASH_AUTO_MAX_BATCHED_TOKENS"
  elif (( memory >= 40000 )); then
    batched_tokens=65536
  elif [[ "$NATIVE_GROUP_PRECISION" == fp8 && "$tensor_parallel_size" == 2 ]]; then
    batched_tokens=2048
  else
    batched_tokens=8192
  fi
  VLLM_MAX_NUM_BATCHED_TOKENS="$batched_tokens"
  NOMP_SIDECAR_MIN_BUFFERED_PROOFS="$(( start > 4 ? start / 2 : 2 ))"
  if (( memory >= 40000 )); then
    # A high-VRAM profile starts with 1024 in-flight jobs, so it needs enough
    # completed-proof room to absorb a local submit burst as well. 2048 proofs
    # is roughly 320 MiB at the current proof size and prevents the 512-proof
    # low-VRAM limit from making min and max equal on this profile.
    NOMP_SIDECAR_MAX_BUFFERED_PROOFS=2048
  else
    NOMP_SIDECAR_MAX_BUFFERED_PROOFS="$(( cap * 2 ))"
    (( NOMP_SIDECAR_MAX_BUFFERED_PROOFS <= 512 )) || NOMP_SIDECAR_MAX_BUFFERED_PROOFS=512
  fi
  (( NOMP_SIDECAR_MIN_BUFFERED_PROOFS < NOMP_SIDECAR_MAX_BUFFERED_PROOFS )) || \
    NOMP_SIDECAR_MIN_BUFFERED_PROOFS="$(( NOMP_SIDECAR_MAX_BUFFERED_PROOFS / 2 ))"
  required_buffer="$(( cap + prefetch ))"
  (( required_buffer <= NOMP_SIDECAR_MAX_BUFFERED_PROOFS )) || required_buffer="$NOMP_SIDECAR_MAX_BUFFERED_PROOFS"
  (( NOMP_SIDECAR_MAX_BUFFERED_PROOFS >= required_buffer )) || \
    fail "Native auto proof buffer cannot cover the configured NOMP_SIDECAR_PREFETCH_REQUESTS."
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

as_root() {
  if ((EUID == 0)); then
    "$@"
  else
    require_command sudo
    sudo "$@"
  fi
}

install_system_packages() {
  [[ "${TENSORCASH_NATIVE_SKIP_APT:-false}" =~ ^(1|true|yes)$ ]] && return 0
  require_command apt-get
  echo "Installing native TensorCash build/runtime dependencies..."
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential cmake git curl wget unzip rsync patch pkg-config ccache \
    nasm yasm libtool autoconf automake m4 \
    libboost-all-dev libflint-dev libgmp-dev libzmq3-dev \
    libssl-dev libcrypto++-dev libargon2-dev libargon2-1 ca-certificates
}

ensure_native_rsync() {
  command -v rsync >/dev/null 2>&1 && return 0

  [[ "${TENSORCASH_NATIVE_SKIP_APT:-false}" =~ ^(1|true|yes)$ ]] && \
    fail "Missing required command: rsync. Automatic package installation is disabled by TENSORCASH_NATIVE_SKIP_APT."
  require_command apt-get

  # `rsync` is needed to overlay the launcher-managed sidecar on every native
  # start, including a cached runtime. Do not reject a minimal HiveOS/rental
  # image before its normal bootstrap gets the chance to install it.
  echo "Installing missing native launcher dependency: rsync..."
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends rsync
  require_command rsync
}

ensure_blackwell_cuda_toolkit() {
  local cuda_home distro keyring_url keyring_deb
  [[ "$NATIVE_PROFILE" == blackwell ]] || return 0
  if cuda_home="$(blackwell_cuda_home 2>/dev/null)"; then
    export CUDA_HOME="$cuda_home"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    return 0
  fi

  [[ "${TENSORCASH_BLACKWELL_AUTO_INSTALL_CUDA_TOOLKIT:-true}" =~ ^(1|true|yes)$ ]] || \
    fail "RTX 50-series native build needs CUDA Toolkit 13. Set TENSORCASH_BLACKWELL_CUDA_HOME or allow automatic toolkit installation."
  [[ -r /etc/os-release ]] || fail "Cannot determine Linux distribution for CUDA Toolkit 13 installation."
  # shellcheck disable=SC1091
  source /etc/os-release
  case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04) distro=ubuntu2204 ;;
    ubuntu:24.04) distro=ubuntu2404 ;;
    *) fail "Automatic CUDA Toolkit 13 installation supports Ubuntu 22.04/24.04 only; install nvcc 13 manually and set TENSORCASH_BLACKWELL_CUDA_HOME." ;;
  esac
  echo "Installing CUDA Toolkit 13 for native Blackwell vLLM compilation (one-time download)..."
  keyring_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb"
  keyring_deb="/tmp/tensorcash-cuda-keyring.deb"
  as_root wget -q -O "$keyring_deb" "$keyring_url" || fail "Could not download NVIDIA CUDA repository keyring."
  as_root dpkg -i "$keyring_deb"
  rm -f "$keyring_deb"
  as_root apt-get update
  as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cuda-toolkit-13-0
  cuda_home="$(blackwell_cuda_home 2>/dev/null)" || \
    fail "CUDA Toolkit 13 installation completed but nvcc was not found. Set TENSORCASH_BLACKWELL_CUDA_HOME explicitly."
  export CUDA_HOME="$cuda_home"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

native_runtime_ld_library_path() {
  local torch_library_path
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME/lib64" ]] || \
      fail "Native Blackwell runtime lost its CUDA 13 library path after bootstrap."
    # The vLLM extension is compiled against the libtorch shipped in this
    # venv.  Do not let a host-level PyTorch/libc10 in /usr/local/lib win the
    # dynamic-linker search: that produces an apparently inexplicable
    # MessageLogger undefined-symbol error despite a freshly rebuilt wheel.
    torch_library_path="$(blackwell_torch_library_path)"
    printf '%s\n' "$torch_library_path:$CUDA_HOME/lib64:/usr/local/lib:/usr/lib/x86_64-linux-gnu"
    return 0
  fi
  printf '%s\n' '/usr/local/lib:/usr/lib/x86_64-linux-gnu'
}

blackwell_torch_library_path() {
  local torch_library_path
  [[ "$NATIVE_PROFILE" == blackwell ]] || return 0
  torch_library_path="$($NATIVE_PY - <<'PY'
from pathlib import Path
import torch

library_path = Path(torch.__file__).resolve().parent / "lib"
if not library_path.is_dir():
    raise RuntimeError(f"PyTorch library directory is missing: {library_path}")
print(library_path)
PY
)"
  [[ "$torch_library_path" == "$NATIVE_VENV"/* && -d "$torch_library_path" ]] || \
    fail "Native Blackwell PyTorch libraries are not inside the managed venv: $torch_library_path"
  printf '%s\n' "$torch_library_path"
}

sync_tensorcash_source() {
  if [[ ! -d "$NATIVE_SOURCE/.git" ]] || [[ "$(git -C "$NATIVE_SOURCE" rev-parse HEAD 2>/dev/null || true)" != "$TENSORCASH_SOURCE_REF" ]]; then
    echo "Fetching public TensorCash source at $TENSORCASH_SOURCE_REF..."
    rm -rf "$NATIVE_SOURCE"
    mkdir -p "$NATIVE_SOURCE"
    git -C "$NATIVE_SOURCE" init -q
    git -C "$NATIVE_SOURCE" remote add origin "$TENSORCASH_SOURCE_URL"
    git -C "$NATIVE_SOURCE" fetch -q --depth 1 origin "$TENSORCASH_SOURCE_REF"
    git -C "$NATIVE_SOURCE" checkout -q --detach FETCH_HEAD
  fi
  [[ "$(git -C "$NATIVE_SOURCE" rev-parse HEAD)" == "$TENSORCASH_SOURCE_REF" ]] || fail "TensorCash source ref did not resolve to the pinned commit."

  # vllm-v010 is a pinned Git submodule, not a directory contained in the
  # parent commit.  A plain shallow fetch leaves an empty mount point and then
  # fails later at rsync.  Fetch only this required submodule; bcore/llama and
  # the other large submodules are not part of native mining startup.
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    # The Blackwell v0.19 tree is deliberately fetched exactly as the OCI
    # builder does. Its gitlink is separate from v0.10 and we do not let an
    # old submodule checkout select an arbitrary revision.
    if [[ ! -d "$NATIVE_VLLM_SOURCE/.git" ]] || [[ "$(git -C "$NATIVE_VLLM_SOURCE" rev-parse HEAD 2>/dev/null || true)" != "$TENSORCASH_BLACKWELL_VLLM_REF" ]]; then
      rm -rf "$NATIVE_VLLM_SOURCE"
      mkdir -p "$NATIVE_VLLM_SOURCE"
      git -C "$NATIVE_VLLM_SOURCE" init -q
      git -C "$NATIVE_VLLM_SOURCE" remote add origin "$TENSORCASH_BLACKWELL_VLLM_URL"
      git -C "$NATIVE_VLLM_SOURCE" fetch -q --depth 1 origin "$TENSORCASH_BLACKWELL_VLLM_REF"
      git -C "$NATIVE_VLLM_SOURCE" checkout -q --detach FETCH_HEAD
    fi
    [[ "$(git -C "$NATIVE_VLLM_SOURCE" rev-parse HEAD)" == "$TENSORCASH_BLACKWELL_VLLM_REF" ]] || \
      fail "TensorCash Blackwell vLLM source ref did not resolve to the pinned commit."
    [[ -d "$NATIVE_VLLM_SOURCE/vllm" ]] || fail "TensorCash Blackwell vLLM source is incomplete."
  else
    git -C "$NATIVE_SOURCE" submodule sync -- services/miner-api/vllm-v010
    git -C "$NATIVE_SOURCE" submodule update --init --depth 1 services/miner-api/vllm-v010
    [[ -d "$NATIVE_VLLM_SOURCE/vllm" ]] || \
      fail "TensorCash vLLM v0.10 submodule is unavailable after checkout."
  fi
}

prepare_blackwell_python_runtime() {
  local build_source proxy_requirements requirements_dir requirements_file
  local wheel_dir wheel_marker wheel torch_index torch_abi build_jobs runtime_ld_library_path
  ensure_blackwell_cuda_toolkit
  if [[ ! -x "$NATIVE_PY" ]]; then
    python3.10 -m venv "$NATIVE_VENV"
  fi
  "$NATIVE_PY" -m pip install --upgrade pip wheel 'setuptools>=77,<81' setuptools-scm \
    cmake ninja packaging pybind11 'numpy>=2' 'scipy>=1.13'

  # CUDA 13 torch and the v0.19 C++ extensions must come from the same ABI
  # family. Do not install PyPI's generic CUDA-12 vLLM wheel here: it would
  # import beside the CUDA-13 torch wheel and fail on its first allocation.
  torch_index="$TENSORCASH_BLACKWELL_TORCH_INDEX_URL"
  echo "Installing CUDA 13 PyTorch $TENSORCASH_BLACKWELL_TORCH_VERSION for native Blackwell..."
  "$NATIVE_PY" -m pip install --pre --upgrade --force-reinstall --no-cache-dir \
    --index-url "$torch_index" \
    "torch==$TENSORCASH_BLACKWELL_TORCH_VERSION" \
    "torchvision==$TENSORCASH_BLACKWELL_TORCHVISION_VERSION" \
    "torchaudio==$TENSORCASH_BLACKWELL_TORCHAUDIO_VERSION"

  # Resolve all Python dependencies before deciding whether the compiled vLLM
  # extension can be reused.  Installing them after the wheel leaves a window
  # where pip can alter libtorch and make vllm/_C.abi3.so unloadable even when
  # the public torch version string is unchanged.
  requirements_dir="$NATIVE_BUILD/blackwell-requirements"
  rm -rf "$requirements_dir"
  mkdir -p "$requirements_dir"
  cp -a "$NATIVE_VLLM_SOURCE/requirements/." "$requirements_dir/"
  requirements_file="$requirements_dir/cuda.txt"
  sed -i.bak -E '/^(torch|torchvision|torchaudio)[[:space:]=]/d' \
    "$requirements_file"
  rm -f "$requirements_file.bak"
  "$NATIVE_PY" -m pip install --no-cache-dir -r "$requirements_file"
  # vLLM 0.19 requires NumPy 2; the legacy proxy pin would otherwise silently
  # downgrade it after the source build and make the CUDA extension unloadable.
  proxy_requirements="$NATIVE_BUILD/blackwell-proxy-requirements.txt"
  sed -E '/^numpy==/d' "$NATIVE_SOURCE/services/miner-api/proxy_requirements.txt" > "$proxy_requirements"
  "$NATIVE_PY" -m pip install --no-cache-dir -r "$proxy_requirements"

  # A PyTorch version alone is not an ABI identity for a locally compiled
  # extension: CUDA/nightly rebuilds can retain the same public version while
  # changing c10 symbols.  Bind the cached Blackwell vLLM wheel to the actual
  # libtorch build fingerprint, not merely to `torch==2.10.0`.
  torch_abi="$($NATIVE_PY - <<'PY'
import hashlib
import torch

payload = "\n".join((
    torch.__version__,
    str(torch.version.cuda),
    str(getattr(torch.version, "git_version", "")),
    str(torch.compiled_with_cxx11_abi()),
))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
PY
)"
  [[ "$torch_abi" =~ ^[a-f0-9]{64}$ ]] || fail "Could not determine native Blackwell PyTorch ABI fingerprint."
  echo "Native Blackwell PyTorch ABI fingerprint: ${torch_abi:0:16}..."

  wheel_dir="$NATIVE_BUILD/blackwell-vllm-wheels"
  wheel_marker="$wheel_dir/.built-from"
  mkdir -p "$wheel_dir"
  wheel="$(find "$wheel_dir" -maxdepth 1 -type f -name 'vllm-*.whl' -print -quit 2>/dev/null || true)"
  if [[ ! -n "$wheel" ]] || ! grep -Fxq "vllm_ref=$TENSORCASH_BLACKWELL_VLLM_REF" "$wheel_marker" 2>/dev/null || \
      ! grep -Fxq "torch=$TENSORCASH_BLACKWELL_TORCH_VERSION" "$wheel_marker" 2>/dev/null || \
      ! grep -Fxq "torch_abi=$torch_abi" "$wheel_marker" 2>/dev/null; then
    echo "Building TensorCash Blackwell vLLM locally for sm_120; this is a one-time CUDA compilation and may take several hours..."
    rm -f "$wheel_dir"/vllm-*.whl "$wheel_marker"
    # Build isolation would obey the source's strict torch requirement by
    # downloading another wheel. Reuse the CUDA-13 torch just installed.
    # Keep the pinned source immutable: the work tree is a disposable build
    # snapshot, so a failed/retried compile always starts from clean inputs.
    build_source="$NATIVE_BUILD/blackwell-vllm-source"
    rm -rf "$build_source"
    mkdir -p "$build_source"
    rsync -a --delete --exclude='.git' "$NATIVE_VLLM_SOURCE/" "$build_source/"
    git -C "$build_source" init -q
    git -C "$build_source" config user.email 'native-build@tensorcash.local'
    git -C "$build_source" config user.name 'TensorCash Native Build'
    git -C "$build_source" add -A
    git -C "$build_source" commit -qm 'Blackwell vLLM build snapshot'
    git -C "$build_source" tag -a v0.19.0 -m 'TensorCash Blackwell build snapshot'
    sed -i.bak -E '/^(torch|torchvision|torchaudio)[[:space:]=]/d' \
      "$build_source/requirements/build.txt" \
      "$build_source/requirements/cuda.txt"
    build_jobs="${TENSORCASH_BLACKWELL_BUILD_JOBS:-2}"
    positive_integer "$build_jobs" TENSORCASH_BLACKWELL_BUILD_JOBS
    (( build_jobs <= 8 )) || fail "TENSORCASH_BLACKWELL_BUILD_JOBS must not exceed 8."
    TORCH_CUDA_ARCH_LIST='12.0;12.0+PTX' \
      VLLM_TARGET_DEVICE=cuda \
      MAX_JOBS="$build_jobs" \
      CCACHE_DIR="$NATIVE_BUILD/blackwell-ccache" \
      VLLM_INSTALL_PUNICA_KERNELS=0 \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM='0.19.0+pow' \
      CUDA_HOME="$CUDA_HOME" \
      "$NATIVE_PY" -m pip wheel --no-deps --no-build-isolation -v \
        "$build_source" -w "$wheel_dir"
    wheel="$(find "$wheel_dir" -maxdepth 1 -type f -name 'vllm-*.whl' -print -quit)"
    [[ -n "$wheel" ]] || fail "Native Blackwell vLLM build did not produce a wheel."
    {
      printf 'vllm_ref=%s\n' "$TENSORCASH_BLACKWELL_VLLM_REF"
      printf 'torch=%s\n' "$TENSORCASH_BLACKWELL_TORCH_VERSION"
      printf 'torch_abi=%s\n' "$torch_abi"
      printf 'cuda_home=%s\n' "$CUDA_HOME"
      printf 'built_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$wheel_marker"
    chmod 600 "$wheel_marker"
  fi

  "$NATIVE_PY" -m pip install --force-reinstall --no-deps "$wheel"
  runtime_ld_library_path="$(native_runtime_ld_library_path)"
  LD_LIBRARY_PATH="$runtime_ld_library_path" "$NATIVE_PY" - <<'PY'
import torch
import vllm

assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
arches = set(torch.cuda.get_arch_list())
assert any(arch.startswith("sm_120") for arch in arches), arches
print("Native TensorCash Blackwell runtime build: OK", torch.__version__, sorted(arches))
PY
}

prepare_python_runtime() {
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    prepare_blackwell_python_runtime
    return 0
  fi
  if [[ ! -x "$NATIVE_PY" ]]; then
    python3.10 -m venv "$NATIVE_VENV"
  fi
  "$NATIVE_PY" -m pip install --upgrade pip wheel 'setuptools<81'
  "$NATIVE_PY" -m pip install --no-cache-dir -r "$NATIVE_SOURCE/services/miner-api/requirements_v10.txt"
  "$NATIVE_PY" -m pip install --no-cache-dir -r "$NATIVE_SOURCE/services/miner-api/proxy_requirements.txt"
  "$NATIVE_PY" -m pip install --no-cache-dir numpy pybind11
}

repair_stock_vllm_if_needed() {
  local site_packages flash_layers
  site_packages="$($NATIVE_PY -c 'import site; print(site.getsitepackages()[0])')"
  flash_layers="$site_packages/vllm/vllm_flash_attn/layers"

  # TensorCash carries a pure-Python fork over the stock vLLM 0.10 wheel.  The
  # wheel itself owns vllm_flash_attn and vllm._version; neither exists in the
  # fork.  Older native-launcher builds used rsync --delete and removed them.
  # Restore only the pinned wheel (without re-resolving all CUDA dependencies)
  # before applying the overlay, so rerunning the launcher repairs that state.
  if [[ ! -d "$flash_layers" ]] || [[ ! -f "$site_packages/vllm/_version.py" ]]; then
    echo "Repairing the stock vLLM 0.10 wheel files required by FlashAttention..."
    "$NATIVE_PY" -m pip install --no-cache-dir --force-reinstall --no-deps 'vllm==0.10.0'
  fi
  [[ -d "$flash_layers" && -f "$site_packages/vllm/_version.py" ]] || \
    fail "The vLLM 0.10 wheel is missing vllm_flash_attn after repair."
}

build_chiavdf() {
  local source_dir="$NATIVE_SOURCE/shared-utils/chiavdf"
  local work_dir="$NATIVE_BUILD/chiavdf"
  local wheel_dir="$NATIVE_BUILD/chiavdf-wheels"
  if "$NATIVE_PY" -c 'import chiavdf' >/dev/null 2>&1; then
    return 0
  fi
  echo "Building native ChiaVDF with assembly support..."
  rm -rf "$work_dir"
  mkdir -p "$work_dir" "$wheel_dir"
  rsync -a --delete "$source_dir/" "$work_dir/"
  git -C "$work_dir" init -q
  git -C "$work_dir" config user.email 'native-build@tensorcash.local'
  git -C "$work_dir" config user.name 'TensorCash Native Build'
  git -C "$work_dir" add .
  git -C "$work_dir" commit -qm 'Native build source'
  git -C "$work_dir" tag -a v1.0.0 -m 'Version 1.0.0'
  GMP_USE_ASM=1 FLINT_ENABLE_ASM=1 CHIAVDF_NO_ASM='' BUILD_VDF_CLIENT=N \
    PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig \
    "$NATIVE_PY" -m pip wheel "$work_dir" -w "$wheel_dir"
  "$NATIVE_PY" -m pip install "$wheel_dir"/*.whl
  "$NATIVE_PY" -c 'import chiavdf; print("chiavdf native build: OK")'
}

build_proof_processor() {
  local pow_dir="$NATIVE_SOURCE/shared-utils/pow-utils"
  local output="$pow_dir/tests/build/proof_processor.so"
  local site_packages
  site_packages="$($NATIVE_PY -c 'import site; print(site.getsitepackages()[0])')"
  if [[ ! -f "$output" ]]; then
    echo "Building TensorCash C++ proof processor..."
    if [[ ! -f /usr/include/zmq.hpp ]]; then
      as_root wget -q https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp -O /usr/include/zmq.hpp
    fi
    PYTHON_EXEC="$NATIVE_PY" FB_SCHEMAS_DIR="$NATIVE_SOURCE/shared-utils/fb-schemas" \
      bash "$pow_dir/tests/build_proofprocessor_simple.sh"
  fi
  [[ -f "$output" ]] || fail "Proof processor compilation did not produce proof_processor.so."
  install -m 755 "$output" "$site_packages/proof_processor.so"
}

prepare_blackwell_python_sources() {
  local site_packages generated_python pow_helper_path
  site_packages="$($NATIVE_PY -c 'import site; print(site.getsitepackages()[0])')"
  generated_python="$NATIVE_SOURCE/shared-utils/pow-utils/tests/build/generated-python"
  [[ -d "$generated_python/proof" ]] || fail "Generated FlatBuffer proof modules are missing."
  [[ -d "$NATIVE_VLLM_SOURCE/vllm" ]] || fail "TensorCash Blackwell vLLM source is missing its Python overlay."

  # Keep the proof sampler deterministic at the float32 CDF boundary. The
  # parent TensorCash source is immutable at SOURCE_REF, so this patch is
  # deterministic and remains idempotent across ordinary launcher restarts.
  if ! grep -Fq 'batch_sample_tokens requires a non-empty [B, V] CDF' \
    "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py"; then
    patch --batch --fuzz=2 -d "$NATIVE_SOURCE" -p1 < "$script_dir/native-cdf-tail.patch"
  fi
  grep -Fq 'batch_sample_tokens requires a non-empty [B, V] CDF' \
    "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" || \
    fail "Native CDF boundary patch was not installed."

  echo "Installing TensorCash Blackwell vLLM/proxy overlays..."
  # The locally-built wheel owns vLLM's CUDA extensions. Overlay only the
  # TensorCash Python code, deliberately preserving every compiled _C*.so.
  rsync -a --exclude='_C*.so' --exclude='*.so.*' \
    "$NATIVE_VLLM_SOURCE/vllm/" "$site_packages/vllm/"
  mkdir -p "$site_packages/vllm/sampling/proof"
  rsync -a --delete "$generated_python/proof/" "$site_packages/vllm/sampling/proof/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/common_sampler_helper.py" \
    "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" \
    "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_v3.py" \
    "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/bcred_table_r1024.py" \
    "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/zmq_pow_writer.py" \
    "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/uint256_arithmetics.py" \
    "$site_packages/vllm/sampling/"
  pow_helper_path="$site_packages/vllm/sampling/common_sampler_helper.py"
  [[ -f "$pow_helper_path" ]] || fail "Native Blackwell PoW sampler helper was not installed."

  rm -rf "$NATIVE_PROXY"
  mkdir -p "$NATIVE_PROXY"
  rsync -a "$NATIVE_SOURCE/services/miner-api/src/" "$NATIVE_PROXY/"
  mkdir -p "$NATIVE_PROXY/utils" "$NATIVE_PROXY/config" "$NATIVE_PROXY/proof"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_v3.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/bcred_table_r1024.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/uint256_arithmetics.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/config/constants.py" "$NATIVE_PROXY/config/"
  rsync -a "$generated_python/proof/" "$NATIVE_PROXY/proof/"
  install -m 644 "$script_dir/nomp-sidecar-overlay.py" "$NATIVE_PROXY/components/nomp_sidecar.py"
  install -m 644 "$script_dir/sidecar-status-overlay.py" "$NATIVE_PROXY/sitecustomize.py"
  grep -Fq 'NOMP sidecar dropped duplicate proof' "$NATIVE_PROXY/components/nomp_sidecar.py" || \
    fail "Native NOMP sidecar overlay is missing concurrent-proof de-duplication."
  grep -Fq 'Released superseded TensorCash VDF prover' "$NATIVE_PROXY/sitecustomize.py" || \
    fail "Native NOMP sidecar overlay is missing VDF memory release handling."
  "$NATIVE_PY" -m py_compile "$NATIVE_PROXY/components/nomp_sidecar.py"
  patch --batch --fuzz=2 -d "$NATIVE_PROXY" -p1 < "$script_dir/native-nomp-proxy.patch"
  grep -Fq "app.router.add_post('/v1/tensorcash/jobs'" "$NATIVE_PROXY/main.py" || \
    fail "Native NOMP route patch did not install /v1/tensorcash/jobs."
  grep -Fq "app.router.add_get('/v1/tensorcash/metrics', self.nomp_sidecar.metrics)" "$NATIVE_PROXY/main.py" || \
    fail "Native NOMP route patch did not install /v1/tensorcash/metrics."
  grep -Fq "NOMP_SIDECAR_ENABLED" "$NATIVE_PROXY/components/constants.py" || \
    fail "Native NOMP constants patch was not installed."
}

prepare_python_sources() {
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    prepare_blackwell_python_sources
    return 0
  fi
  local site_packages generated_python
  site_packages="$($NATIVE_PY -c 'import site; print(site.getsitepackages()[0])')"
  generated_python="$NATIVE_SOURCE/shared-utils/pow-utils/tests/build/generated-python"
  [[ -d "$generated_python/proof" ]] || fail "Generated FlatBuffer proof modules are missing."

  # Keep PoW CDF sampling inside [0, V-1] even when float32 cumsum rounds its
  # terminal boundary below one. The public source ref is intentionally pinned,
  # so this small, audited patch is deterministic and idempotent across resume.
  if ! grep -Fq 'batch_sample_tokens requires a non-empty [B, V] CDF' \
    "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py"; then
    patch --batch --fuzz=2 -d "$NATIVE_SOURCE" -p1 < "$script_dir/native-cdf-tail.patch"
  fi
  grep -Fq 'batch_sample_tokens requires a non-empty [B, V] CDF' \
    "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" || \
    fail "Native CDF boundary patch was not installed."

  # The public TensorCash sampler used a fixed 1024-row proof ring. A vLLM
  # sample batch can include its bounded waiting reserve as well, so high
  # profiles need the row pool to follow POW_MAX_CONCURRENCY.
  local pow_sampler_path
  pow_sampler_path="$NATIVE_SOURCE/services/miner-api/vllm-v010/vllm/v1/sample/ops/topk_topp_sampler.py"
  if ! grep -Fq 'TensorCash row pool must cover every possible vLLM sample row' \
    "$pow_sampler_path"; then
    patch --batch --fuzz=2 -d "$NATIVE_SOURCE" -p1 < "$script_dir/native-pow-row-capacity.patch"
  fi
  grep -Fq 'TensorCash row pool must cover every possible vLLM sample row' \
    "$pow_sampler_path" || \
    fail "Native PoW row-capacity patch was not installed."
  # The source version imported os inside __init__. Once the capacity patch
  # reads os before that line, Python correctly treats it as an unbound local.
  # Remove only that redundant inner import. Keeping this as a dedicated,
  # idempotent patch also upgrades runtimes that already received v1 of the
  # row-capacity overlay before this correction.
  if grep -Fq '        import os' "$pow_sampler_path"; then
    patch --batch --fuzz=2 -d "$NATIVE_SOURCE" -p1 < "$script_dir/native-pow-row-import-fix.patch"
  fi
  grep -Fq '        import os' "$pow_sampler_path" && \
    fail "Native PoW row-capacity overlay retained a conflicting inner os import."

  local pow_helper_path
  pow_helper_path="$NATIVE_SOURCE/shared-utils/pow-utils/common_sampler_helper.py"
  if ! grep -Fq 'TensorCash protects current sample rows during row allocation' \
    "$pow_helper_path"; then
    patch --batch --fuzz=2 -d "$NATIVE_SOURCE" -p1 < "$script_dir/native-pow-row-protection.patch"
  fi
  grep -Fq 'TensorCash protects current sample rows during row allocation' \
    "$pow_helper_path" || \
    fail "Native PoW row-protection patch was not installed."

  echo "Installing TensorCash vLLM/proxy overlays..."
  repair_stock_vllm_if_needed
  # Deliberately do not add --delete here. TensorCash replaces pure Python
  # files only; the upstream wheel retains native FlashAttention files and
  # vllm._version that are not present in the TensorCash fork.
  rsync -a --exclude='*.so' "$NATIVE_SOURCE/services/miner-api/vllm-v010/vllm/" "$site_packages/vllm/"
  mkdir -p "$site_packages/vllm/sampling/proof"
  rsync -a --delete "$generated_python/proof/" "$site_packages/vllm/sampling/proof/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/common_sampler_helper.py" "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_v3.py" "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/bcred_table_r1024.py" "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/zmq_pow_writer.py" "$site_packages/vllm/sampling/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/uint256_arithmetics.py" "$site_packages/vllm/sampling/"

  rm -rf "$NATIVE_PROXY"
  mkdir -p "$NATIVE_PROXY"
  rsync -a "$NATIVE_SOURCE/services/miner-api/src/" "$NATIVE_PROXY/"
  mkdir -p "$NATIVE_PROXY/utils" "$NATIVE_PROXY/config" "$NATIVE_PROXY/proof"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_utils.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/pow_v3.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/bcred_table_r1024.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/pow-utils/uint256_arithmetics.py" "$NATIVE_PROXY/utils/"
  install -m 644 "$NATIVE_SOURCE/shared-utils/config/constants.py" "$NATIVE_PROXY/config/"
  rsync -a "$generated_python/proof/" "$NATIVE_PROXY/proof/"
  install -m 644 "$script_dir/nomp-sidecar-overlay.py" "$NATIVE_PROXY/components/nomp_sidecar.py"
  install -m 644 "$script_dir/sidecar-status-overlay.py" "$NATIVE_PROXY/sitecustomize.py"
  # Native mode deliberately re-installs this public scheduler overlay on each
  # ordinary launcher run. This makes controller-side fixes (including proof
  # de-duplication under concurrent inference) take effect without rebuilding
  # the Python environment or downloading the model again.
  grep -Fq 'NOMP sidecar dropped duplicate proof' "$NATIVE_PROXY/components/nomp_sidecar.py" || \
    fail "Native NOMP sidecar overlay is missing concurrent-proof de-duplication."
  grep -Fq 'Released superseded TensorCash VDF prover' "$NATIVE_PROXY/sitecustomize.py" || \
    fail "Native NOMP sidecar overlay is missing VDF memory release handling."
  "$NATIVE_PY" -m py_compile "$NATIVE_PROXY/components/nomp_sidecar.py"
  # The public TensorCash source is intentionally NOMP-agnostic.  Apply the
  # small, audited integration patch after copying it so the native proxy has
  # the same authenticated local work-unit routes as the Docker runtime.
  # The official pinned main.py has a generated line-number offset in its
  # upstream diff metadata. The content is immutable at SOURCE_REF; fuzz=2
  # accepts that harmless offset while patch still rejects missing contexts.
  patch --batch --fuzz=2 -d "$NATIVE_PROXY" -p1 < "$script_dir/native-nomp-proxy.patch"
  grep -Fq "app.router.add_post('/v1/tensorcash/jobs'" "$NATIVE_PROXY/main.py" || \
    fail "Native NOMP route patch did not install /v1/tensorcash/jobs."
  grep -Fq "app.router.add_get('/v1/tensorcash/metrics', self.nomp_sidecar.metrics)" "$NATIVE_PROXY/main.py" || \
    fail "Native NOMP route patch did not install /v1/tensorcash/metrics."
  grep -Fq "NOMP_SIDECAR_ENABLED" "$NATIVE_PROXY/components/constants.py" || \
    fail "Native NOMP constants patch was not installed."
}

runtime_marker_is_current() {
  if [[ "$NATIVE_PROFILE" == blackwell ]]; then
    local runtime_ld_library_path
    runtime_ld_library_path="$(native_runtime_ld_library_path 2>/dev/null)" || return 1
    [[ -f "$NATIVE_MARKER" ]] && \
      grep -Fxq 'profile=blackwell' "$NATIVE_MARKER" && \
      grep -Fxq "source_ref=$TENSORCASH_SOURCE_REF" "$NATIVE_MARKER" && \
      grep -Fxq "vllm_ref=$TENSORCASH_BLACKWELL_VLLM_REF" "$NATIVE_MARKER" && \
      grep -Fxq "torch=$TENSORCASH_BLACKWELL_TORCH_VERSION" "$NATIVE_MARKER" && \
      [[ -x "$NATIVE_VLLM" ]] && [[ -f "$NATIVE_PROXY/main.py" ]] && \
      LD_LIBRARY_PATH="$runtime_ld_library_path" "$NATIVE_PY" - <<'PY' >/dev/null 2>&1
import chiavdf
import proof_processor
import torch
import vllm
assert torch.cuda.is_available()
assert any(arch.startswith("sm_120") for arch in torch.cuda.get_arch_list())
PY
    return
  fi
  [[ -f "$NATIVE_MARKER" ]] && grep -Fxq "source_ref=$TENSORCASH_SOURCE_REF" "$NATIVE_MARKER" && \
    [[ -x "$NATIVE_VLLM" ]] && [[ -f "$NATIVE_PROXY/main.py" ]] && \
    "$NATIVE_PY" -c 'import chiavdf, proof_processor, vllm; from vllm.vllm_flash_attn.layers import rotary' >/dev/null 2>&1
}

bootstrap_runtime() {
  mkdir -p "$NATIVE_HOME" "$NATIVE_BUILD" "$NATIVE_INSTANCES" "$NATIVE_HOME/capacity"
  chmod 700 "$NATIVE_HOME" "$NATIVE_INSTANCES" "$NATIVE_HOME/capacity"
  # A minimal rental container may lack wget/ca-certificates before we add the
  # NVIDIA CUDA repository. Install the ordinary build prerequisites first
  # only when CUDA 13 is actually absent; cached Blackwell starts do not pay
  # this apt step again.
  if [[ "$NATIVE_PROFILE" == blackwell ]] && ! blackwell_cuda_home >/dev/null 2>&1; then
    install_system_packages
  fi
  # The marker import exercises the compiled vLLM extension too, so establish
  # the CUDA-13 runtime path before deciding whether a cached Blackwell wheel
  # is healthy. This is a no-op for legacy profiles.
  ensure_blackwell_cuda_toolkit
  if ! "$rebuild" && runtime_marker_is_current; then
    # Keep the launcher-owned sidecar scheduler in sync on every launch.  The
    # expensive wheel/extension build remains cached, but a scheduler update
    # must not wait for miners to rebuild their native environment.
    sync_tensorcash_source
    prepare_python_sources
    echo "Using existing native TensorCash runtime: $NATIVE_HOME"
    return 0
  fi
  install_system_packages
  sync_tensorcash_source
  prepare_python_runtime
  build_chiavdf
  build_proof_processor
  prepare_python_sources
  "$NATIVE_PY" - <<'PY'
import chiavdf
import proof_processor
import torch
import vllm
assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
print("Native TensorCash runtime self-test: OK", torch.cuda.get_device_name(0))
PY
  {
    printf 'profile=%s\n' "$NATIVE_PROFILE"
    printf 'source_ref=%s\n' "$TENSORCASH_SOURCE_REF"
    if [[ "$NATIVE_PROFILE" == blackwell ]]; then
      printf 'vllm_ref=%s\n' "$TENSORCASH_BLACKWELL_VLLM_REF"
      printf 'torch=%s\n' "$TENSORCASH_BLACKWELL_TORCH_VERSION"
    fi
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$NATIVE_MARKER"
  chmod 600 "$NATIVE_MARKER"
}

model_cache_name() {
  printf '%s' "${MODEL_NAME//\//--}"
}

download_model() {
  local cache_name snapshot marker attempts delay attempt
  cache_name="$(model_cache_name)"
  snapshot="$MODELS_DATA/hub/models--${cache_name}/snapshots/${MODEL_COMMIT}"
  marker="$MODELS_DATA/.tensorcash-model-${cache_name}-${MODEL_COMMIT}.complete"
  attempts="${TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS:-12}"
  delay="${TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS:-15}"
  positive_integer "$attempts" TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS
  positive_integer "$delay" TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS
  mkdir -p "$MODELS_DATA" "$RUNTIME_DATA"
  chmod 755 "$MODELS_DATA"
  chmod 700 "$RUNTIME_DATA"
  if [[ -f "$marker" && -f "$snapshot/config.json" ]] && compgen -G "$snapshot/*.safetensors" >/dev/null; then
    NATIVE_MODEL_SNAPSHOT="$snapshot"
    return 0
  fi
  for ((attempt = 1; attempt <= attempts; attempt += 1)); do
    echo "Downloading ${MODEL_NAME}@${MODEL_COMMIT} (attempt $attempt/$attempts; cache resumes after interruption)..."
    if MODEL_NAME="$MODEL_NAME" MODEL_COMMIT="$MODEL_COMMIT" MODELS_DATA="$MODELS_DATA" \
      HTTP_PROXY="${TENSORCASH_HTTP_PROXY:-${HTTP_PROXY:-}}" HTTPS_PROXY="${TENSORCASH_HTTP_PROXY:-${HTTPS_PROXY:-}}" \
      "$NATIVE_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    revision=os.environ["MODEL_COMMIT"],
    cache_dir=os.path.join(os.environ["MODELS_DATA"], "hub"),
)
PY
    then
      break
    fi
    (( attempt < attempts )) || fail "Pinned model download failed after $attempts attempts."
    echo "Model download interrupted; retrying in ${delay}s with the existing cache..." >&2
    sleep "$delay"
  done
  [[ -f "$snapshot/config.json" ]] || fail "Model cache lacks config.json after download."
  compgen -G "$snapshot/*.safetensors" >/dev/null || fail "Model cache lacks safetensors weights after download."
  {
    printf 'model=%s\n' "$MODEL_NAME"
    printf 'commit=%s\n' "$MODEL_COMMIT"
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$marker"
  chmod 600 "$marker"
  NATIVE_MODEL_SNAPSHOT="$snapshot"
}

native_static_fp8_candidate_exists() {
  local index count memory static_min fp8_min
  static_min="$(tensorcash_static_fp8_tp1_min_vram_mib)"
  fp8_min="$(tensorcash_fp8_single_min_vram_mib)"
  count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || return 1
  for ((index = 0; index < count; index += 1)); do
    if [[ "$NATIVE_PROFILE" == blackwell ]] && ! gpu_is_blackwell "$index"; then
      continue
    fi
    memory="$(nvidia-smi --id="$index" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "$memory" =~ ^[1-9][0-9]*$ ]] || continue
    (( memory >= static_min && memory < fp8_min )) && return 0
  done
  return 1
}

configure_native_static_fp8_snapshot() {
  local configured snapshot models_root
  TENSORCASH_STATIC_FP8_TP1_AVAILABLE=false
  NATIVE_STATIC_FP8_SNAPSHOT=""
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
    "$models_root"/*) ;;
    *) fail "TENSORCASH_STATIC_FP8_SNAPSHOT must stay below MODELS_DATA." ;;
  esac
  TENSORCASH_STATIC_FP8_TP1_AVAILABLE=true
  NATIVE_STATIC_FP8_SNAPSHOT="$snapshot"
  export TENSORCASH_STATIC_FP8_TP1_AVAILABLE
  echo "Validated official serialized FP8 snapshot for native 12 GiB TP=1: $snapshot"
}

download_native_static_fp8_model() {
  local repository commit cache_name snapshot marker attempts delay attempt
  repository="${TENSORCASH_STATIC_FP8_REPOSITORY:-Qwen/Qwen3-8B-FP8}"
  commit="${TENSORCASH_STATIC_FP8_COMMIT:-220b46e3b2180893580a4454f21f22d3ebb187d3}"
  [[ "$repository" == Qwen/Qwen3-8B-FP8 && "$commit" == 220b46e3b2180893580a4454f21f22d3ebb187d3 ]] || \
    fail "The 12 GiB profile requires the tested Qwen/Qwen3-8B-FP8@220b46e3b2180893580a4454f21f22d3ebb187d3 artifact."
  cache_name="${repository//\//--}"
  snapshot="$MODELS_DATA/hub/models--${cache_name}/snapshots/${commit}"
  marker="$snapshot/.tensorcash-static-fp8.complete"
  attempts="${TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS:-12}"
  delay="${TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS:-15}"
  positive_integer "$attempts" TENSORCASH_MODEL_DOWNLOAD_ATTEMPTS
  positive_integer "$delay" TENSORCASH_MODEL_DOWNLOAD_DELAY_SECONDS
  mkdir -p "$MODELS_DATA" "$RUNTIME_DATA"
  chmod 755 "$MODELS_DATA"

  if [[ ! -f "$marker" ]]; then
    for ((attempt = 1; attempt <= attempts; attempt += 1)); do
      echo "Downloading official serialized FP8 Qwen3-8B (attempt $attempt/$attempts; cache resumes after interruption)..."
      if MODEL_NAME="$repository" MODEL_COMMIT="$commit" MODELS_DATA="$MODELS_DATA" \
        HTTP_PROXY="${TENSORCASH_HTTP_PROXY:-${HTTP_PROXY:-}}" HTTPS_PROXY="${TENSORCASH_HTTP_PROXY:-${HTTPS_PROXY:-}}" \
        "$NATIVE_PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ["MODEL_NAME"],
    revision=os.environ["MODEL_COMMIT"],
    cache_dir=os.path.join(os.environ["MODELS_DATA"], "hub"),
)
PY
      then
        break
      fi
      (( attempt < attempts )) || fail "Serialized FP8 model download failed after $attempts attempts."
      echo "Static FP8 model download interrupted; retrying in ${delay}s with the existing cache..." >&2
      sleep "$delay"
    done
    [[ -f "$snapshot/config.json" ]] || fail "Serialized FP8 cache lacks config.json after download."
    compgen -G "$snapshot/*.safetensors" >/dev/null || fail "Serialized FP8 cache lacks safetensors weights after download."
    grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"fp8"' "$snapshot/config.json" || \
      fail "Downloaded Qwen3-8B-FP8 artifact lacks the required FP8 quantization config."
    find "$snapshot" -type d -exec chmod a+rx {} +
    find "$snapshot" -type f -exec chmod a+r {} +
    umask 077
    printf 'format=tensorcash-static-fp8-v1\nmodel=%s\ncommit=%s\nartifact_repository=%s\nartifact_commit=%s\ncompleted_utc=%s\n' \
      "$MODEL_NAME" "$MODEL_COMMIT" "$repository" "$commit" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$marker"
  fi
  export TENSORCASH_STATIC_FP8_SNAPSHOT="$snapshot"
  configure_native_static_fp8_snapshot
}

prepare_native_static_fp8_profile() {
  if [[ "$TENSORCASH_MODEL_PRECISION" != bf16 ]] && native_static_fp8_candidate_exists; then
    download_native_static_fp8_model
  elif [[ -n "${TENSORCASH_STATIC_FP8_SNAPSHOT:-}" ]]; then
    configure_native_static_fp8_snapshot
  fi
}

native_requires_canonical_model() {
  local group
  local -a groups=()
  mapfile -t groups < <(resolve_native_gpu_groups)
  for group in "${groups[@]}"; do
    native_group_profile "$group"
    [[ "$NATIVE_GROUP_USES_STATIC_FP8" == true ]] || return 0
  done
  return 1
}

pid_file() { printf '%s/%s.pid\n' "$NATIVE_PIDS" "$1"; }

pid_running() {
  local file
  file="$(pid_file "$1")"
  [[ -f "$file" ]] || return 1
  kill -0 "$(<"$file")" 2>/dev/null
}

stop_vllm_attempt_group() {
  local file="$NATIVE_PIDS/vllm-attempt.pid" attempt waited=0
  [[ -r "$file" ]] || return 0
  IFS= read -r attempt < "$file" || true
  [[ "$attempt" =~ ^[1-9][0-9]*$ ]] || { rm -f "$file"; return 0; }
  if kill -0 -- "-$attempt" 2>/dev/null; then
    echo "Stopping native vLLM attempt process group (PGID $attempt)..."
    kill -TERM -- "-$attempt" 2>/dev/null || true
    while kill -0 -- "-$attempt" 2>/dev/null && (( waited < 20 )); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 -- "-$attempt" 2>/dev/null; then
      echo "Native vLLM attempt PGID $attempt ignored TERM; sending KILL." >&2
      kill -KILL -- "-$attempt" 2>/dev/null || true
    fi
  fi
  rm -f "$file"
}

stop_process() {
  local name="$1" file pid
  file="$(pid_file "$name")"
  # The vLLM bootstrap creates a nested setsid group for its TP workers.
  # Stop it first; killing only this outer shell can otherwise leave an
  # unowned CUDA context that no container PID can subsequently release.
  [[ "$name" == vllm ]] && stop_vllm_attempt_group
  [[ -f "$file" ]] || return 0
  pid="$(<"$file")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "Stopping native $name (PID $pid)..."
    kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$file"
}

stop_instance() {
  stop_process miner
  stop_process proxy
  stop_process vllm
}

legacy_instance_paths() {
  NATIVE_INSTANCE='legacy'
  NATIVE_INSTANCE_HOME="$NATIVE_HOME"
  NATIVE_LOGS="$NATIVE_HOME/logs"
  NATIVE_PIDS="$NATIVE_HOME/pids"
  NATIVE_INSTANCE_ENV="$NATIVE_HOME/runtime.env"
}

stop_all() {
  local instance
  native_paths
  # Native versions before multi-GPU support stored a single group's files
  # directly under runtime/native. Stop it too during migration.
  legacy_instance_paths
  stop_instance
  for instance in "$NATIVE_INSTANCES"/g*; do
    [[ -d "$instance" ]] || continue
    native_instance_paths "$(basename "$instance")"
    stop_instance
  done
}

show_instance_status() {
  local name file pid sidecar_port
  printf '=== native TensorCash %s ===\n' "$NATIVE_INSTANCE"
  for name in vllm proxy miner; do
    file="$(pid_file "$name")"
    if pid_running "$name"; then
      pid="$(<"$file")"
      printf '%-6s running (pid %s)\n' "$name" "$pid"
    else
      printf '%-6s stopped\n' "$name"
    fi
  done
  sidecar_port="$(sed -n 's/^HTTP_PORT=//p' "$NATIVE_INSTANCE_ENV" 2>/dev/null | tail -n 1)"
  [[ "$sidecar_port" =~ ^[0-9]+$ ]] || sidecar_port=8080
  printf 'sidecar '
  curl -fsS "http://127.0.0.1:${sidecar_port}/health" 2>/dev/null || echo 'unavailable'
}

show_native_status() {
  native_paths
  local instance found=false
  for instance in "$NATIVE_INSTANCES"/g*; do
    [[ -d "$instance" ]] || continue
    found=true
    native_instance_paths "$(basename "$instance")"
    show_instance_status
  done
  if ! "$found"; then
    legacy_instance_paths
    show_instance_status
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== GPU ==="
    nvidia-smi --query-gpu=index,name,pstate,power.draw,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader
  fi
}

show_native_plan() {
  local group number=0 capacity_file checkpoint_kind
  local -a gpu_groups=()
  mapfile -t gpu_groups < <(resolve_native_gpu_groups)
  echo "=== native TensorCash launch plan ==="
  for group in "${gpu_groups[@]}"; do
    number=$((number + 1))
    native_group_profile "$group"
    checkpoint_kind=canonical
    [[ "$NATIVE_GROUP_USES_STATIC_FP8" == true ]] && checkpoint_kind=serialized-fp8
    capacity_file="$(native_capacity_file "$NATIVE_GROUP_GPU_NAME" "$NATIVE_GROUP_MIN_MEMORY" "${TENSORCASH_AUTO_CONCURRENCY_CEILING:-1024}" "$NATIVE_GROUP_PRECISION" "$NATIVE_GROUP_TENSOR_PARALLEL_SIZE" "$MAX_MODEL_LEN" "$GPU_MEM_UTIL" "$checkpoint_kind")"
    printf 'g%s: GPUs %s (%s, min %s MiB, TP=%s, %s, %s) -> vLLM=%s sidecar=%s proof=%s capacity=%s\n' \
      "$number" "$NATIVE_GROUP_GPU_IDS" "$NATIVE_GROUP_GPU_NAME" "$NATIVE_GROUP_MIN_MEMORY" "$NATIVE_GROUP_TENSOR_PARALLEL_SIZE" "$NATIVE_GROUP_PRECISION" "$checkpoint_kind" "$((8000 + number - 1))" \
      "$((8080 + number - 1))" "$((7002 + number - 1))" "$capacity_file"
  done
}

wait_for_http() {
  local url="$1" seconds="$2" title="$3"
  local elapsed=0
  while (( elapsed < seconds )); do
    curl -fsS "$url" >/dev/null 2>&1 && return 0
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "=== $title log ===" >&2
  tail -n 100 "$NATIVE_LOGS/${title}.log" >&2 || true
  fail "$title did not become ready within ${seconds}s."
}

wait_for_vllm_capacity() {
  local file="$1" seconds="$2" requested="$3"
  local elapsed=0 value
  while (( elapsed < seconds )); do
    if [[ -r "$file" ]]; then
      IFS= read -r value < "$file" || true
      if [[ "${value:-}" =~ ^[1-9][0-9]*$ ]] && (( value <= requested )); then
        printf '%s\n' "$value"
        return 0
      fi
    fi
    if ! pid_running vllm; then
      echo '=== vllm log ===' >&2
      tail -n 100 "$NATIVE_LOGS/vllm.log" >&2 || true
      fail 'Native vLLM exited before its capacity probe completed.'
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo '=== vllm log ===' >&2
  tail -n 100 "$NATIVE_LOGS/vllm.log" >&2 || true
  fail "Native vLLM did not finish its capacity probe within ${seconds}s."
}

native_instance_token() {
  local instance="$1"
  printf '%s' "${NOMP_SIDECAR_TOKEN}:${instance}" | sha256sum | awk '{print $1}'
}

seed_profile_capacity_from_legacy() {
  local destination="$1" minimum="$2" legacy="$RUNTIME_DATA/native/vllm-effective-max-seqs" value
  [[ ! -r "$destination" && -r "$legacy" ]] || return 0
  IFS= read -r value < "$legacy" || return 0
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || return 0
  (( value >= minimum && value <= VLLM_MAX_NUM_SEQS )) || return 0
  mkdir -p "$(dirname "$destination")"
  umask 077
  printf '%s\n' "$value" > "$destination"
  echo "Reusing legacy GPU0 bootstrap-confirmed max sequences=$value for this matching hardware profile."
}

start_native_instance() {
  local instance_number="$1" gpu_group="$2"
  local memory gpu_name precision quantization tensor_parallel_size env_file vllm_effective_file vllm_fallback_min effective_max_seqs
  local instance_data vllm_port sidecar_port collector_port sidecar_token group_worker
  local pow_row_capacity prefetch_for_rows runtime_ld_library_path cpu_affinity
  local cache_name instance_model_snapshot checkpoint_kind
  native_instance_paths "g$instance_number"
  native_group_profile "$gpu_group"
  memory="$NATIVE_GROUP_MIN_MEMORY"
  gpu_name="$NATIVE_GROUP_GPU_NAME"
  precision="$NATIVE_GROUP_PRECISION"
  quantization="$NATIVE_GROUP_QUANTIZATION"
  tensor_parallel_size="$NATIVE_GROUP_TENSOR_PARALLEL_SIZE"
  cpu_affinity="$(native_group_cpu_affinity "$gpu_group" || true)"
  checkpoint_kind=canonical
  instance_model_snapshot="$NATIVE_MODEL_SNAPSHOT"
  if [[ "$NATIVE_GROUP_USES_STATIC_FP8" == true ]]; then
    checkpoint_kind=serialized-fp8
    instance_model_snapshot="$NATIVE_STATIC_FP8_SNAPSHOT"
  fi
  [[ -n "$instance_model_snapshot" && -f "$instance_model_snapshot/config.json" ]] || \
    fail "Native GPU group $gpu_group has no validated model snapshot."
  configure_native_memory_profile "$memory" "$tensor_parallel_size" "$precision"
  tensorcash_validate_gpu_mem_util "$precision" "$NATIVE_GROUP_GPU_MEM_UTIL" || \
    fail "GPU_MEM_UTIL is not valid for ${precision} on GPU group $gpu_group."
  if [[ "$TENSORCASH_CONCURRENCY_MODE" == auto ]]; then
    configure_native_auto_concurrency "$memory" "$tensor_parallel_size"
    echo "Native $NATIVE_INSTANCE auto concurrency: start=${NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY}, cap=${VLLM_MAX_NUM_SEQS}, step=${NOMP_SIDECAR_ADAPTIVE_STEP}, prefetch=${NOMP_SIDECAR_PREFETCH_REQUESTS}, context=${NATIVE_GROUP_MAX_MODEL_LEN}, gpu_mem_util=${NATIVE_GROUP_GPU_MEM_UTIL}"
  fi
  if [[ "$TENSORCASH_CONCURRENCY_MODE" == manual ]]; then
    (( VLLM_MAX_NUM_SEQS >= NOMP_SIDECAR_CONCURRENCY )) || fail "VLLM_MAX_NUM_SEQS must cover NOMP_SIDECAR_CONCURRENCY."
  fi
  prefetch_for_rows="${NOMP_SIDECAR_PREFETCH_REQUESTS:-0}"
  [[ "$prefetch_for_rows" =~ ^[0-9]+$ ]] || \
    fail "NOMP_SIDECAR_PREFETCH_REQUESTS must be numeric before starting native vLLM."
  pow_row_capacity="$(( VLLM_MAX_NUM_SEQS + prefetch_for_rows ))"
  (( pow_row_capacity >= 1 && pow_row_capacity <= 4096 )) || \
    fail "Native PoW row capacity must be between 1 and 4096."
  vllm_effective_file="$(native_capacity_file "$gpu_name" "$memory" "$VLLM_MAX_NUM_SEQS" "$precision" "$tensor_parallel_size" "$NATIVE_GROUP_MAX_MODEL_LEN" "$NATIVE_GROUP_GPU_MEM_UTIL" "$checkpoint_kind")"
  vllm_fallback_min="$VLLM_MAX_NUM_SEQS"
  if [[ "$TENSORCASH_CONCURRENCY_MODE" == auto ]]; then
    vllm_fallback_min="$NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY"
  fi
  # The single-GPU native layout used this location before profile-keyed
  # capacity reuse existed. Preserve a valid first-card discovery across the
  # launcher upgrade so a same-model multi-card host does not pay it twice.
  if (( instance_number == 1 && tensor_parallel_size == 1 )) && [[ "$precision" == bf16 && "$gpu_group" == 0 ]]; then
    seed_profile_capacity_from_legacy "$vllm_effective_file" "$vllm_fallback_min"
  fi
  cache_name="$(model_cache_name)"
  instance_data="$RUNTIME_DATA/native/$NATIVE_INSTANCE"
  native_allocate_instance_ports "$instance_number"
  vllm_port="$NATIVE_VLLM_PORT"
  sidecar_port="$NATIVE_SIDECAR_PORT"
  collector_port="$NATIVE_COLLECTOR_PORT"
  sidecar_token="$(native_instance_token "$NATIVE_INSTANCE")"
  group_worker="${WORKER}-${NATIVE_INSTANCE}"
  runtime_ld_library_path="$(native_runtime_ld_library_path)"

  mkdir -p "$NATIVE_LOGS" "$NATIVE_PIDS" "$instance_data"
  chmod 700 "$NATIVE_INSTANCE_HOME" "$NATIVE_LOGS" "$NATIVE_PIDS" "$instance_data"
  env_file="$NATIVE_INSTANCE_ENV"
  umask 077
  cat > "$env_file" <<EOF
PYTHONPATH=$NATIVE_PROXY
LD_LIBRARY_PATH=$runtime_ld_library_path
CUDA_HOME=${CUDA_HOME:-}
TENSORCASH_NATIVE_RUNTIME_PROFILE=$NATIVE_PROFILE
TENSORCASH_NATIVE_CPU_AFFINITY=$cpu_affinity
CUDA_VISIBLE_DEVICES=$gpu_group
MODEL_NAME=$MODEL_NAME
MODEL_COMMIT=$MODEL_COMMIT
MODEL_DIFFICULTY_NORMALIZER=$MODEL_DIFFICULTY_NORMALIZER
MAX_MODEL_LEN=$NATIVE_GROUP_MAX_MODEL_LEN
GPU_MEM_UTIL=$NATIVE_GROUP_GPU_MEM_UTIL
TENSORCASH_MODEL_PRECISION=$precision
VLLM_TENSOR_PARALLEL_SIZE=$tensor_parallel_size
VLLM_MAX_NUM_SEQS=$VLLM_MAX_NUM_SEQS
POW_MAX_CONCURRENCY=$pow_row_capacity
TENSORCASH_VLLM_QUANTIZATION=$quantization
VLLM_PORT=$vllm_port
VLLM_CUDA_GRAPH_SIZES=${VLLM_CUDA_GRAPH_SIZES:-}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-}
VLLM_MODEL_PATH=$instance_model_snapshot
CHAT_TEMPLATE_PATH=$NATIVE_SOURCE/deployments/simple-worker/chat-template/qwen3.5-enhanced.jinja
TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE=$vllm_effective_file
TENSORCASH_VLLM_RUNTIME_CAPACITY_FILE=$vllm_effective_file
TENSORCASH_VLLM_ATTEMPT_PID_FILE=$NATIVE_PIDS/vllm-attempt.pid
TENSORCASH_VLLM_FALLBACK_MIN_SEQS=$vllm_fallback_min
TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS=${TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS:-900}
TENSORCASH_VLLM_RUNTIME_RECOVERY_STEP=${TENSORCASH_VLLM_RUNTIME_RECOVERY_STEP:-64}
TENSORCASH_VLLM_RUNTIME_RECOVERY_BACKOFF_SECONDS=${TENSORCASH_VLLM_RUNTIME_RECOVERY_BACKOFF_SECONDS:-10}
NOMP_SIDECAR_TOKEN=$sidecar_token
NOMP_SIDECAR_ENABLED=true
NOMP_SIDECAR_CONCURRENCY=$NOMP_SIDECAR_CONCURRENCY
NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY=${NOMP_SIDECAR_ADAPTIVE_START_CONCURRENCY:-}
NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY=${NOMP_SIDECAR_ADAPTIVE_MAX_CONCURRENCY:-}
NOMP_SIDECAR_ADAPTIVE_STEP=${NOMP_SIDECAR_ADAPTIVE_STEP:-}
NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS=$NOMP_SIDECAR_ADAPTIVE_CONFIRM_WINDOWS
NOMP_SIDECAR_MIN_BUFFERED_PROOFS=$NOMP_SIDECAR_MIN_BUFFERED_PROOFS
NOMP_SIDECAR_MAX_BUFFERED_PROOFS=$NOMP_SIDECAR_MAX_BUFFERED_PROOFS
TENSORCASH_SUBMIT_WINDOW=$TENSORCASH_SUBMIT_WINDOW
MINING_SOLUTION_COOLDOWN_SEC=0
WORKER_MODE=standalone
STANDALONE_MODE=true
HTTP_HOST=127.0.0.1
HTTP_PORT=$sidecar_port
TARGET_URL=http://127.0.0.1:$vllm_port
API_KEY=internal-secret
TEST_MODE=false
PRIORITY_MODE=false
MIN_ACTIVE_REQUESTS=0
MINING_ENABLED=true
POW_PROOF_VERSION=3
PROOF_CACHE_ENABLED=false
PROOF_COLLECTOR_PORT=$collector_port
TOOL_CALL_PARSER=qwen3_coder
USE_VLLM_XARGS=false
HF_HOME=$MODELS_DATA
HF_HUB_CACHE=$MODELS_DATA/hub
TRANSFORMERS_CACHE=$MODELS_DATA/hub
HF_MODULES_CACHE=$instance_data/hf-modules
# Persist graph compilation artifacts per native GPU group.  Both locations
# participate in the vLLM/Torch compilation path and are safe to reuse only
# within this instance's model, precision, TP and capacity profile.
VLLM_CACHE_ROOT=$instance_data/vllm-cache
TORCHINDUCTOR_CACHE_DIR=$instance_data/torchinductor-cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_TELEMETRY=1
VLLM_ENABLE_POW=1
POW_EGRESS_MODE=broker
POW_PROXY_ENABLE=false
ZMQ_PUSH_HOST=127.0.0.1
ZMQ_PUSH_PORT=$collector_port
POW_PROCESSOR_MODE=cpp
VLLM_ENABLE_RESPONSES_API_STORE=1
VLLM_DO_NOT_TRACK=1
TENSORCASH_VLLM_CLEANUP_GPU_IDS=$gpu_group
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}
EOF
  if [[ -n "${NOMP_SIDECAR_ADMISSION_SPREAD_MS:-}" ]]; then
    printf 'NOMP_SIDECAR_ADMISSION_SPREAD_MS=%s\n' \
      "$NOMP_SIDECAR_ADMISSION_SPREAD_MS" >> "$env_file"
  fi
  if [[ -n "${NOMP_SIDECAR_PREFETCH_REQUESTS:-}" ]]; then
    printf 'NOMP_SIDECAR_PREFETCH_REQUESTS=%s\n' \
      "$NOMP_SIDECAR_PREFETCH_REQUESTS" >> "$env_file"
  fi
  chmod 600 "$env_file"

  echo "Starting native TensorCash $NATIVE_INSTANCE on GPUs $gpu_group (${gpu_name}, min ${memory} MiB, TP=${tensor_parallel_size}, ${precision}), requested max sequences=$VLLM_MAX_NUM_SEQS, cpu_affinity=${cpu_affinity:-unbound}..."
  (
    set -a
    source "$env_file"
    set +a
    export VLLM_BIN="$NATIVE_VLLM"
    export VLLM_HOST=127.0.0.1
    export VLLM_PORT="$vllm_port"
    export VLLM_HEALTH_URL="http://127.0.0.1:${vllm_port}/health"
    export VLLM_CDF_PATCH_PATH=''
    export VLLM_BOOT_LOG=''
    if [[ -n "$cpu_affinity" ]]; then
      exec taskset -c "$cpu_affinity" bash "$script_dir/vllm-local-cache.sh"
    fi
    exec bash "$script_dir/vllm-local-cache.sh"
  ) >"$NATIVE_LOGS/vllm.log" 2>&1 &
  echo $! > "$(pid_file vllm)"
  effective_max_seqs="$(wait_for_vllm_capacity \
    "$vllm_effective_file" 1800 "$VLLM_MAX_NUM_SEQS")"
  wait_for_http "http://127.0.0.1:${vllm_port}/health" 120 vllm
  printf '\n# Written by the vLLM bootstrap capacity probe.\nVLLM_MAX_NUM_SEQS=%s\n' \
    "$effective_max_seqs" >> "$env_file"
  echo "Native $NATIVE_INSTANCE vLLM bootstrap-confirmed max sequences=$effective_max_seqs"

  echo "Starting native TensorCash sidecar..."
  (
    set -a
    source "$env_file"
    set +a
    cd "$NATIVE_PROXY"
    if [[ -n "$cpu_affinity" ]]; then
      exec taskset -c "$cpu_affinity" "$NATIVE_PY" main.py
    fi
    exec "$NATIVE_PY" main.py
  ) >"$NATIVE_LOGS/proxy.log" 2>&1 &
  echo $! > "$(pid_file proxy)"
  wait_for_http "http://127.0.0.1:${sidecar_port}/health" 300 proxy

  echo "Starting TensorCash controller..."
  (
    set -a
    source "$env_file"
    set +a
    if [[ -n "$cpu_affinity" ]]; then
      exec taskset -c "$cpu_affinity" "$script_dir/runtime/bin/niuquanminer" \
        --algo tensorcash --pool "$POOL_HOST" --port "$POOL_PORT" \
        --wallet "$PAYOUT_ACCOUNT" --worker "$group_worker" \
        --tensorcash-sidecar "http://127.0.0.1:${sidecar_port}" \
        --tensorcash-sidecar-token "$sidecar_token" \
        --tensorcash-poll-ms "$TENSORCASH_POLL_MS" \
        --tensorcash-submit-window "$TENSORCASH_SUBMIT_WINDOW" \
        --stats-interval "$TENSORCASH_STATS_INTERVAL"
    fi
    exec "$script_dir/runtime/bin/niuquanminer" \
      --algo tensorcash --pool "$POOL_HOST" --port "$POOL_PORT" \
      --wallet "$PAYOUT_ACCOUNT" --worker "$group_worker" \
      --tensorcash-sidecar "http://127.0.0.1:${sidecar_port}" \
      --tensorcash-sidecar-token "$sidecar_token" \
      --tensorcash-poll-ms "$TENSORCASH_POLL_MS" \
      --tensorcash-submit-window "$TENSORCASH_SUBMIT_WINDOW" \
      --stats-interval "$TENSORCASH_STATS_INTERVAL"
  ) >"$NATIVE_LOGS/miner.log" 2>&1 &
  echo $! > "$(pid_file miner)"
  sleep 2
  pid_running miner || { tail -n 100 "$NATIVE_LOGS/miner.log" >&2 || true; fail "TensorCash controller exited immediately."; }
  echo "Native TensorCash $NATIVE_INSTANCE started: GPUs $gpu_group, TP=${tensor_parallel_size}, vLLM=$vllm_port, sidecar=$sidecar_port."
  echo "Logs: $NATIVE_LOGS/{vllm,proxy,miner}.log"
}

start_native() {
  local group number=0
  local -a gpu_groups=()
  mapfile -t gpu_groups < <(resolve_native_gpu_groups)
  echo "Native TensorCash selected GPU groups: $(IFS=';'; printf '%s' "${gpu_groups[*]}")"
  stop_all
  NATIVE_RESERVED_PORTS=''
  for group in "${gpu_groups[@]}"; do
    number=$((number + 1))
    start_native_instance "$number" "$group"
  done
  echo "Native TensorCash started ${#gpu_groups[@]} group(s)."
}

while (($#)); do
  case "$1" in
    --pool) pool_arg="${2:-}"; shift 2 ;;
    --wallet) wallet_arg="${2:-}"; shift 2 ;;
    --worker) worker_arg="${2:-}"; shift 2 ;;
    --gpu) gpu_arg="${2:-}"; shift 2 ;;
    --install) install_only=true; shift ;;
    --rebuild-runtime) rebuild=true; shift ;;
    --stop) stop_only=true; shift ;;
    --status) status_only=true; shift ;;
    --plan) plan_only=true; shift ;;
    --logs) logs_only=true; shift ;;
    --purge-runtime) purge_only=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

if "$stop_only"; then
  stop_all
  exit 0
fi
if "$purge_only"; then
  native_paths
  stop_all
  rm -rf "$NATIVE_HOME"
  echo "Deleted script-owned native runtime: $NATIVE_HOME"
  exit 0
fi
if "$logs_only"; then
  log_files=()
  native_paths
  for instance_dir in "$NATIVE_INSTANCES"/g*; do
    [[ -d "$instance_dir/logs" ]] || continue
    log_files+=("$instance_dir/logs/miner.log" "$instance_dir/logs/proxy.log" "$instance_dir/logs/vllm.log")
  done
  if ((${#log_files[@]} == 0)); then
    # Preserve access to logs from the pre-multi-GPU native layout.
    mkdir -p "$NATIVE_HOME/logs"
    log_files=("$NATIVE_HOME/logs/miner.log" "$NATIVE_HOME/logs/proxy.log" "$NATIVE_HOME/logs/vllm.log")
  fi
  tail -n 100 -f "${log_files[@]}"
  exit 0
fi
if "$status_only"; then
  show_native_status
  exit 0
fi

require_command bash
require_command curl
require_command git
require_command nvidia-smi
require_command sha256sum
load_config
ensure_native_rsync
ensure_native_open_file_limit
native_paths
if "$plan_only"; then
  show_native_plan
  exit 0
fi
bootstrap_runtime
ensure_compatible_miner_binary
if "$install_only"; then
  echo "Native TensorCash runtime is ready. Start it with: bash native-vast.sh"
  exit 0
fi
prepare_native_static_fp8_profile
if native_requires_canonical_model; then
  download_model
else
  echo "All selected native GPU groups use the verified serialized FP8 snapshot; skipping the BF16 model download."
fi
start_native
