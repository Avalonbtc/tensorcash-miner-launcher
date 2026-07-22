#!/usr/bin/env bash
# The runtime image's original entry point accepts a Hugging Face repository
# name.  In offline mining deployments, make vLLM load the already-verified
# local snapshot instead while retaining the canonical served model name for
# TensorCash proof metadata.
set -euo pipefail

: "${MODEL_NAME:=Qwen/Qwen3-8B}"
: "${MAX_MODEL_LEN:=2048}"
: "${DEVICE:=auto}"
: "${GPU_MEM_UTIL:=0.78}"
: "${VLLM_TENSOR_PARALLEL_SIZE:=1}"
: "${VLLM_MAX_NUM_SEQS:=1}"
: "${VLLM_CUDA_GRAPH_SIZES:=}"
: "${API_KEY:=internal-secret}"
: "${TOOL_CALL_PARSER:=qwen3_coder}"
: "${CHAT_TEMPLATE_PATH:=/opt/chat-template/qwen3.5-enhanced.jinja}"
: "${VLLM_BIN:=vllm}"
: "${VLLM_HOST:=0.0.0.0}"
: "${VLLM_PORT:=8000}"
: "${VLLM_HEALTH_URL:=http://127.0.0.1:${VLLM_PORT}/health}"
: "${TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE:=/data/vllm-effective-max-seqs}"
: "${TENSORCASH_VLLM_FALLBACK_MIN_SEQS:=32}"
: "${TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS:=900}"
# A failed vLLM server can leave TP worker descendants alive briefly after the
# parent exits.  Never begin another bootstrap until those workers have gone
# and every GPU exposed to this sidecar is back below this idle threshold.
: "${TENSORCASH_VLLM_CLEANUP_TIMEOUT_SECONDS:=120}"
: "${TENSORCASH_VLLM_CLEANUP_MAX_USED_MIB:=512}"
: "${TENSORCASH_VLLM_CLEANUP_GPU_IDS:=}"

# Some hosted Docker daemons do not retain supervisord's /dev/fd/1 child
# output.  Keep an in-container copy so a failed vLLM bootstrap is diagnosable.
boot_log="${VLLM_BOOT_LOG:-/tmp/tensorcash-vllm.log}"
if [[ -n "$boot_log" ]]; then
  exec > >(tee -a "$boot_log") 2>&1
fi

model_path="${VLLM_MODEL_PATH:-$MODEL_NAME}"
if [[ "$model_path" != "$MODEL_NAME" && ! -f "$model_path/config.json" ]]; then
  echo "[vLLM] Local TensorCash model snapshot is incomplete: $model_path/config.json" >&2
  exit 2
fi

export VLLM_ENABLE_POW=1
export POW_EGRESS_MODE=broker
export POW_PROXY_ENABLE=false
export ZMQ_PUSH_HOST=127.0.0.1
export ZMQ_PUSH_PORT="${PROOF_COLLECTOR_PORT:-7002}"
export POW_PROCESSOR_MODE="${POW_PROCESSOR_MODE:-cpp}"
export VLLM_ENABLE_RESPONSES_API_STORE=1

# torch.searchsorted returns V if float32 cumsum leaves the final CDF boundary
# a few ulps below one. Load an import-time wrapper from a read-only mount so
# this fix does not depend on the runtime image allowing writes to site-packages.
if [[ -z "${VLLM_CDF_PATCH_PATH+x}" ]]; then
  VLLM_CDF_PATCH_PATH=/app/vllm-cdf-patch
fi
if [[ -n "$VLLM_CDF_PATCH_PATH" ]]; then
  export PYTHONPATH="$VLLM_CDF_PATCH_PATH${PYTHONPATH:+:${PYTHONPATH}}"
fi

echo "[vLLM] Loading local snapshot: $model_path"
echo "[vLLM] Serving TensorCash model identity: $MODEL_NAME"

positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

requested_max_seqs="$VLLM_MAX_NUM_SEQS"
fallback_min_seqs="$TENSORCASH_VLLM_FALLBACK_MIN_SEQS"
startup_timeout="$TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS"
cleanup_timeout="$TENSORCASH_VLLM_CLEANUP_TIMEOUT_SECONDS"
cleanup_max_used_mib="$TENSORCASH_VLLM_CLEANUP_MAX_USED_MIB"
positive_integer "$requested_max_seqs" || { echo "[vLLM] VLLM_MAX_NUM_SEQS must be a positive integer" >&2; exit 2; }
positive_integer "$fallback_min_seqs" || { echo "[vLLM] TENSORCASH_VLLM_FALLBACK_MIN_SEQS must be a positive integer" >&2; exit 2; }
positive_integer "$startup_timeout" || { echo "[vLLM] TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer" >&2; exit 2; }
positive_integer "$cleanup_timeout" || { echo "[vLLM] TENSORCASH_VLLM_CLEANUP_TIMEOUT_SECONDS must be a positive integer" >&2; exit 2; }
positive_integer "$cleanup_max_used_mib" || { echo "[vLLM] TENSORCASH_VLLM_CLEANUP_MAX_USED_MIB must be a positive integer" >&2; exit 2; }
(( fallback_min_seqs <= requested_max_seqs )) || fallback_min_seqs="$requested_max_seqs"

candidate_max_seqs="$fallback_min_seqs"
saved_max_seqs=""
if [[ -r "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE" ]]; then
  IFS= read -r saved_max_seqs < "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE" || true
  if positive_integer "${saved_max_seqs:-}" && \
      (( saved_max_seqs >= fallback_min_seqs && saved_max_seqs <= requested_max_seqs )); then
    candidate_max_seqs="$saved_max_seqs"
    echo "[vLLM] Reusing previously healthy max-num-seqs=$candidate_max_seqs"
  fi
fi

# The capacity file is also the launcher-visible readiness marker. Keep its
# value long enough to choose a fast restart candidate, then remove it before
# the new server begins booting so Docker cannot declare the sidecar healthy
# while an old capacity value is still on disk.
rm -f "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE"

write_effective_max_seqs() {
  local parent tmp
  parent="$(dirname "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE")"
  mkdir -p "$parent"
  umask 077
  tmp="${TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE}.tmp.$$"
  printf '%s\n' "$1" > "$tmp"
  mv -f "$tmp" "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE"
}

build_args() {
  local max_seqs="$1"
  local graph_csv graph_size
  local -a graph_sizes=()
  local -a applicable_graph_sizes=()
  args=(
    "$VLLM_BIN" serve "$model_path"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
    --max-num-seqs "$max_seqs"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --api-key "$API_KEY"
    --load-format safetensors
    --max-model-len "$MAX_MODEL_LEN"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --chat-template "$CHAT_TEMPLATE_PATH"
    --enable-prompt-tokens-details
  )

  # Explicit capture sizes avoid a first-use CUDA-graph compile stall for the
  # fixed-size mining batches. The launcher supplies a tiered list, but a
  # bootstrap attempt must never request a graph larger than the candidate
  # max-num-seqs it is currently proving can start.
  graph_csv="${VLLM_CUDA_GRAPH_SIZES//[[:space:]]/}"
  if [[ -n "$graph_csv" ]]; then
    IFS=',' read -r -a graph_sizes <<< "$graph_csv"
    for graph_size in "${graph_sizes[@]}"; do
      positive_integer "$graph_size" || {
        echo "[vLLM] VLLM_CUDA_GRAPH_SIZES must be comma-separated positive integers" >&2
        exit 2
      }
      (( graph_size <= max_seqs )) && applicable_graph_sizes+=("$graph_size")
    done
    if ((${#applicable_graph_sizes[@]})); then
      args+=(--cuda-graph-sizes "${applicable_graph_sizes[@]}")
    fi
  fi

  # Keep the chain-pinned commit in vLLM's model configuration even when the
  # weights come from a local snapshot. TensorCash proof metadata reads this
  # field; omitting it turns valid proofs into `model_identifier=...@unknown`.
  if [[ -n "${MODEL_COMMIT:-}" ]]; then
    args+=(--revision "$MODEL_COMMIT")
  fi
  if [[ "$DEVICE" != cpu ]]; then
    args+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
  fi
}

vllm_attempt_pid=""
vllm_attempt_ready=false

visible_gpus_released() {
  # The launch layer provides the physical TP-group indices. This is needed in
  # native mode, where CUDA_VISIBLE_DEVICES does not filter nvidia-smi output.
  # An empty value is retained for standalone/test execution and means all
  # GPUs visible to this process.
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local used gpu
  while IFS= read -r used; do
    used="${used//[[:space:]]/}"
    [[ "$used" =~ ^[0-9]+$ ]] || continue
    if (( used > cleanup_max_used_mib )); then
      return 1
    fi
  done < <(gpu_memory_query memory.used)

  # If nvidia-smi is unavailable or produced no parseable rows, leave the
  # process-group cleanup in place and avoid blocking a non-NVIDIA test host.
  return 0
}

gpu_memory_query() {
  local field="$1" gpu
  local -a cleanup_gpu_ids=()
  if [[ -z "$TENSORCASH_VLLM_CLEANUP_GPU_IDS" ]]; then
    nvidia-smi "--query-gpu=$field" --format=csv,noheader,nounits 2>/dev/null || true
    return 0
  fi
  IFS=',' read -r -a cleanup_gpu_ids <<< "$TENSORCASH_VLLM_CLEANUP_GPU_IDS"
  for gpu in "${cleanup_gpu_ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || continue
    nvidia-smi --id="$gpu" "--query-gpu=$field" --format=csv,noheader,nounits 2>/dev/null || true
  done
}

wait_for_visible_gpus_release() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0

  local elapsed=0
  while (( elapsed < cleanup_timeout )); do
    if visible_gpus_released; then
      # Require two clean samples so CUDA's asynchronous teardown cannot race
      # the next vLLM process and recreate the apparent "not enough VRAM"
      # failure on an otherwise empty TP group.
      sleep 2
      if visible_gpus_released; then
        echo "[vLLM] TP GPU teardown confirmed (${cleanup_max_used_mib} MiB idle threshold)"
        return 0
      fi
    fi

    if (( elapsed == 0 || elapsed % 10 == 0 )); then
      echo "[vLLM] Waiting for previous TP workers to release GPU memory:" >&2
      gpu_memory_query index,memory.used,memory.total >&2
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  echo "[vLLM] TP GPU memory did not return below ${cleanup_max_used_mib} MiB within ${cleanup_timeout}s; refusing an overlapping retry" >&2
  gpu_memory_query index,memory.used,memory.total >&2
  return 1
}

stop_vllm_attempt() {
  local pid="${vllm_attempt_pid:-}" waited=0
  [[ -n "$pid" ]] || return 0

  # `setsid` below makes this PID the process-group leader. vLLM's TP workers
  # inherit the group, so terminating only the parent cannot leave a stale
  # rank occupying a 4070 after an unsuccessful bootstrap.
  if kill -0 -- "-$pid" 2>/dev/null; then
    echo "[vLLM] Stopping failed vLLM process group pgid=$pid"
    kill -TERM -- "-$pid" 2>/dev/null || true
    while kill -0 -- "-$pid" 2>/dev/null && (( waited < 20 )); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      echo "[vLLM] Process group pgid=$pid ignored TERM; sending KILL" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi

  # The parent can already be reaped while TP descendants are still winding
  # down. `wait` handles the parent; the VRAM check below covers descendants.
  wait "$pid" 2>/dev/null || true
  vllm_attempt_pid=""
  wait_for_visible_gpus_release
}

wait_for_vllm_exit() {
  local exit_code=0
  if wait "$vllm_attempt_pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  stop_vllm_attempt || return 70
  return "$exit_code"
}

trap 'stop_vllm_attempt || true' EXIT INT TERM

run_vllm_attempt() {
  local max_seqs="$1" elapsed exit_code=1
  vllm_attempt_ready=false
  build_args "$max_seqs"
  command -v setsid >/dev/null 2>&1 || {
    echo "[vLLM] setsid is required for safe TensorCash TP-worker cleanup" >&2
    return 127
  }
  printf '[vLLM] Bootstrap attempt max-num-seqs=%s:' "$max_seqs"
  printf ' %q' "${args[@]}"
  printf '\n'
  # Every attempt gets its own session/process group. This lets cleanup kill
  # the parent and all TP worker descendants as one unit before any retry.
  setsid "${args[@]}" &
  vllm_attempt_pid=$!
  elapsed=0
  while kill -0 "$vllm_attempt_pid" 2>/dev/null; do
    if curl -fsS --max-time 2 "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
      vllm_attempt_ready=true
      echo "[vLLM] Bootstrap healthy at max-num-seqs=$max_seqs"
      return 0
    fi
    if (( elapsed >= startup_timeout )); then
      echo "[vLLM] Startup timeout at max-num-seqs=$max_seqs; terminating attempt" >&2
      stop_vllm_attempt || return 70
      return 124
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  if wait "$vllm_attempt_pid"; then
    exit_code=1
  else
    exit_code=$?
  fi
  # A vLLM parent can exit before its multiprocessing workers. Always clean
  # the complete session and wait for per-group VRAM to be idle before the
  # supervisor is allowed to retry this script.
  stop_vllm_attempt || return 70
  return "$exit_code"
}

run_final_vllm() {
  local max_seqs="$1" exit_code
  if run_vllm_attempt "$max_seqs"; then
    :
  else
    exit_code=$?
    echo "[vLLM] Final bootstrap failed at max-num-seqs=$max_seqs (exit=$exit_code)" >&2
    return "$exit_code"
  fi
  write_effective_max_seqs "$max_seqs"
  echo "[vLLM] Ready with bootstrap-confirmed max-num-seqs=$max_seqs"
  if wait_for_vllm_exit; then
    return 0
  else
    exit_code=$?
  fi
  return "$exit_code"
}

# vLLM allocates sampler workspace for --max-num-seqs during startup. The
# sidecar itself starts at 32, but handing vLLM 1024 immediately can OOM before
# that scheduler exists. On a fresh runtime, prove capacity in ascending powers
# of two (32, 64, ...); on failure restart the last healthy level. Subsequent
# restarts reuse the recorded value instead of re-running this discovery.
if [[ -n "$saved_max_seqs" ]] && (( candidate_max_seqs == saved_max_seqs )); then
  exec_code=0
  if run_final_vllm "$candidate_max_seqs"; then
    exit 0
  else
    exec_code=$?
  fi
  if [[ "$vllm_attempt_ready" == true ]]; then
    exit "$exec_code"
  fi
  echo "[vLLM] Saved max-num-seqs=$candidate_max_seqs is no longer bootable; rediscovering from $fallback_min_seqs" >&2
  rm -f "$TENSORCASH_VLLM_EFFECTIVE_MAX_SEQS_FILE"
  candidate_max_seqs="$fallback_min_seqs"
fi

last_healthy_max_seqs=0
while true; do
  if run_vllm_attempt "$candidate_max_seqs"; then
    last_healthy_max_seqs="$candidate_max_seqs"
    if (( candidate_max_seqs >= requested_max_seqs )); then
      write_effective_max_seqs "$candidate_max_seqs"
      echo "[vLLM] Ready with bootstrap-confirmed max-num-seqs=$candidate_max_seqs"
      if wait_for_vllm_exit; then
        exit 0
      else
        exit_code=$?
      fi
      exit "$exit_code"
    fi
    next_max_seqs=$((candidate_max_seqs * 2))
    (( next_max_seqs <= requested_max_seqs )) || next_max_seqs="$requested_max_seqs"
    echo "[vLLM] Capacity $candidate_max_seqs is healthy; probing $next_max_seqs before opening the sidecar"
    stop_vllm_attempt
    candidate_max_seqs="$next_max_seqs"
    continue
  else
    exit_code=$?
  fi
  if (( last_healthy_max_seqs == 0 )); then
    echo "[vLLM] Failed even at initial max-num-seqs=$candidate_max_seqs; cannot start TensorCash vLLM" >&2
    exit "$exit_code"
  fi
  echo "[vLLM] Capacity probe $candidate_max_seqs failed (exit=$exit_code); using last healthy $last_healthy_max_seqs" >&2
  final_code=0
  run_final_vllm "$last_healthy_max_seqs" || final_code=$?
  exit "$final_code"
done
