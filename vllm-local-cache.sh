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

# Some hosted Docker daemons do not retain supervisord's /dev/fd/1 child
# output.  Keep an in-container copy so a failed vLLM bootstrap is diagnosable.
boot_log="${VLLM_BOOT_LOG:-/tmp/tensorcash-vllm.log}"
exec > >(tee -a "$boot_log") 2>&1

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
export PYTHONPATH="/app/vllm-cdf-patch${PYTHONPATH:+:${PYTHONPATH}}"

echo "[vLLM] Loading local snapshot: $model_path"
echo "[vLLM] Serving TensorCash model identity: $MODEL_NAME"

args=(
  vllm serve "$model_path"
  --served-model-name "$MODEL_NAME"
  --trust-remote-code
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --host 0.0.0.0
  --port 8000
  --api-key "$API_KEY"
  --load-format safetensors
  --max-model-len "$MAX_MODEL_LEN"
  --enable-auto-tool-choice
  --tool-call-parser "$TOOL_CALL_PARSER"
  --chat-template "$CHAT_TEMPLATE_PATH"
  --enable-prompt-tokens-details
)

# Keep the chain-pinned commit in vLLM's model configuration even when the
# weights come from a local snapshot.  TensorCash proof metadata reads this
# field; omitting it turns valid proofs into `model_identifier=...@unknown`.
if [[ -n "${MODEL_COMMIT:-}" ]]; then
  args+=(--revision "$MODEL_COMMIT")
fi
if [[ "$DEVICE" != cpu ]]; then
  args+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
fi

printf '[vLLM] Executing:'
printf ' %q' "${args[@]}"
printf '\n'
exec "${args[@]}"
