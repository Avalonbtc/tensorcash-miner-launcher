#!/usr/bin/env bash
# Keep the controller CLI contract in the public launcher.  In particular,
# this pinned TensorCash controller does not accept the image's legacy
# --stats-interval flag.
set -euo pipefail

: "${POOL_HOST:?POOL_HOST is required}"
: "${POOL_PORT:?POOL_PORT is required}"
: "${PAYOUT_ACCOUNT:?PAYOUT_ACCOUNT is required}"
: "${WORKER:?WORKER is required}"
: "${NOMP_SIDECAR_TOKEN:?NOMP_SIDECAR_TOKEN is required}"
: "${TENSORCASH_POLL_MS:=200}"
: "${TENSORCASH_SUBMIT_WINDOW:=16}"
: "${TENSORCASH_STATS_INTERVAL:=30}"
: "${POOL_TLS:=false}"
: "${POOL_TLS_INSECURE:=false}"

tls_args=()
case "${POOL_TLS,,}" in
  1|true|yes) tls_args+=(--tls) ;;
  0|false|no) ;;
  *) echo "POOL_TLS must be true or false" >&2; exit 2 ;;
esac
case "${POOL_TLS_INSECURE,,}" in
  1|true|yes)
    ((${#tls_args[@]} > 0)) || { echo "POOL_TLS_INSECURE requires POOL_TLS=true" >&2; exit 2; }
    tls_args+=(--tls-insecure)
    ;;
  0|false|no) ;;
  *) echo "POOL_TLS_INSECURE must be true or false" >&2; exit 2 ;;
esac

exec /opt/tensorcash/niuquanminer \
  --algo tensorcash \
  --pool "$POOL_HOST" \
  --port "$POOL_PORT" \
  --wallet "$PAYOUT_ACCOUNT" \
  --worker "$WORKER" \
  --tensorcash-sidecar http://127.0.0.1:8080 \
  --tensorcash-sidecar-token "$NOMP_SIDECAR_TOKEN" \
  --tensorcash-poll-ms "$TENSORCASH_POLL_MS" \
  --tensorcash-submit-window "$TENSORCASH_SUBMIT_WINDOW" \
  --stats-interval "$TENSORCASH_STATS_INTERVAL" \
  "${tls_args[@]}"
