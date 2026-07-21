#!/usr/bin/env bash
# Compatible replacement for the pinned image's start-proxy.sh.  It preserves
# its normal NOMP/broker startup behaviour, adding only the idempotent local
# proof-claim and performance-metrics routes needed by the bounded submit
# pipeline and the miner's GPU-throughput display.
set -e

python3 /app/nomp-claim-route-patch.py

echo "[Miner-Proxy] Waiting for vLLM to be ready at ${TARGET_URL:-http://127.0.0.1:8000}..."
VLLM_URL="${TARGET_URL:-http://127.0.0.1:8000}"
MAX_RETRIES=60
RETRY_COUNT=0
while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
  if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
    echo "[Miner-Proxy] vLLM is ready!"
    break
  fi
  RETRY_COUNT=$((RETRY_COUNT + 1))
  echo "[Miner-Proxy] Waiting for vLLM... ($RETRY_COUNT/$MAX_RETRIES)"
  sleep 5
done

if [ "$RETRY_COUNT" -eq "$MAX_RETRIES" ]; then
  echo "[Miner-Proxy] WARNING: vLLM health check timed out, starting anyway..."
fi

if [ "${MINING_VLLM_ENABLED:-false}" = "true" ]; then
  if [ -z "${MINING_MODEL_NAME:-}" ] || [ -z "${MINING_MODEL_COMMIT:-}" ]; then
    echo "[Miner-Proxy] ERROR: MINING_VLLM_ENABLED=true requires MINING_MODEL_NAME and MINING_MODEL_COMMIT"
    exit 1
  fi
  PRIMARY_MODEL_NAME="${MODEL_NAME:-}"
  PRIMARY_MODEL_COMMIT="${MODEL_COMMIT:-}"
  export MODEL_NAME="${MINING_MODEL_NAME}"
  export MODEL_COMMIT="${MINING_MODEL_COMMIT}"
  if [ -z "${MODEL_ROUTES:-}" ] && [ -n "$PRIMARY_MODEL_NAME" ]; then
    export MODEL_ROUTES="${PRIMARY_MODEL_NAME}@${PRIMARY_MODEL_COMMIT}=${TARGET_URL:-http://127.0.0.1:8000},${MINING_MODEL_NAME}@${MINING_MODEL_COMMIT}=http://127.0.0.1:${MINING_VLLM_PORT:-8001}"
  fi
fi

if [ "${NOMP_SIDECAR_ENABLED:-false}" = "true" ]; then
  if [ "${WORKER_MODE:-standalone}" = "broker" ]; then
    echo "[Miner-Proxy] ERROR: NOMP_SIDECAR_ENABLED requires WORKER_MODE=standalone"
    exit 1
  fi
  if [ ${#NOMP_SIDECAR_TOKEN} -lt 16 ]; then
    echo "[Miner-Proxy] ERROR: NOMP_SIDECAR_TOKEN must be at least 16 characters"
    exit 1
  fi
  if [ -z "${MODEL_NAME:-}" ] || [ -z "${MODEL_COMMIT:-}" ]; then
    echo "[Miner-Proxy] ERROR: NOMP_SIDECAR_ENABLED requires MODEL_NAME and MODEL_COMMIT"
    exit 1
  fi
  if [ "${MODEL_DIFFICULTY_NORMALIZER:-0}" -lt 1 ]; then
    echo "[Miner-Proxy] ERROR: MODEL_DIFFICULTY_NORMALIZER must be a positive consensus value"
    exit 1
  fi
  echo "[Miner-Proxy] NOMP local-sidecar mode enabled; compute broker disabled."
else
  if [ -z "${BROKER_WS_URL:-}" ] || [ -z "${PROVIDER_JWT_TOKEN:-}" ]; then
    echo "[Miner-Proxy] ERROR: broker mode requires BROKER_WS_URL and PROVIDER_JWT_TOKEN"
    exit 1
  fi
fi

echo "[Miner-Proxy] Configuration: WORKER_MODE=${WORKER_MODE:-broker} TARGET_URL=${TARGET_URL:-http://127.0.0.1:8000}"
cd /app/miner-proxy/src
exec python main.py
