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
positive_integer "$requested_max_seqs" || { echo "[vLLM] VLLM_MAX_NUM_SEQS must be a positive integer" >&2; exit 2; }
positive_integer "$fallback_min_seqs" || { echo "[vLLM] TENSORCASH_VLLM_FALLBACK_MIN_SEQS must be a positive integer" >&2; exit 2; }
positive_integer "$startup_timeout" || { echo "[vLLM] TENSORCASH_VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer" >&2; exit 2; }
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
run_vllm_attempt() {
  local max_seqs="$1" elapsed exit_code=1
  vllm_attempt_ready=false
  build_args "$max_seqs"
  printf '[vLLM] Bootstrap attempt max-num-seqs=%s:' "$max_seqs"
  printf ' %q' "${args[@]}"
  printf '\n'
  "${args[@]}" &
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
      kill -TERM "$vllm_attempt_pid" 2>/dev/null || true
      sleep 5
      kill -KILL "$vllm_attempt_pid" 2>/dev/null || true
      break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  if wait "$vllm_attempt_pid"; then
    exit_code=1
  else
    exit_code=$?
  fi
  vllm_attempt_pid=""
  return "$exit_code"
}

stop_vllm_attempt() {
  [[ -n "$vllm_attempt_pid" ]] || return 0
  kill -TERM "$vllm_attempt_pid" 2>/dev/null || true
  if wait "$vllm_attempt_pid"; then
    :
  fi
  vllm_attempt_pid=""
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
  if wait "$vllm_attempt_pid"; then
    return 0
  else
    exit_code=$?
  fi
  vllm_attempt_pid=""
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
      if wait "$vllm_attempt_pid"; then
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
