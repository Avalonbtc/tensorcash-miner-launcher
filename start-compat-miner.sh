#!/usr/bin/env bash
# CPU-only TensorCash controller.  The CUDA/vLLM sidecar owns GPU inference;
# this process only speaks Stratum and forwards jobs/proofs over loopback.
set -euo pipefail

: "${POOL_HOST:?POOL_HOST is required}"
: "${POOL_PORT:?POOL_PORT is required}"
: "${PAYOUT_ACCOUNT:?PAYOUT_ACCOUNT is required}"
: "${WORKER:?WORKER is required}"
: "${NOMP_SIDECAR_TOKEN:?NOMP_SIDECAR_TOKEN is required}"
: "${TENSORCASH_POLL_MS:=200}"

exec /opt/tensorcash/niuquanminer \
  --algo tensorcash \
  --pool "$POOL_HOST" \
  --port "$POOL_PORT" \
  --wallet "$PAYOUT_ACCOUNT" \
  --worker "$WORKER" \
  --tensorcash-sidecar http://127.0.0.1:8080 \
  --tensorcash-sidecar-token "$NOMP_SIDECAR_TOKEN" \
  --tensorcash-poll-ms "$TENSORCASH_POLL_MS"
