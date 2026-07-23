#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for launcher in start.sh native-vast.sh start-miner-overlay.sh; do
  bash -n "$script_dir/$launcher"
done

for launcher in start.sh native-vast.sh; do
  grep -Fq 'normalize_pool_tls_settings' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must validate TLS configuration." >&2
    exit 1
  }
  grep -Fq -- '--tls-insecure' "$script_dir/$launcher" || {
    echo "FAIL: $launcher must expose the explicit TLS test escape hatch." >&2
    exit 1
  }
done

grep -Fq 'POOL_TLS: ${POOL_TLS:-false}' "$script_dir/docker-compose.yml" || {
  echo 'FAIL: Docker miner must receive POOL_TLS.' >&2
  exit 1
}
grep -Fq 'tls_args+=(--tls)' "$script_dir/start-miner-overlay.sh" || {
  echo 'FAIL: Docker controller must receive --tls.' >&2
  exit 1
}
grep -Fq 'controller_tls_args+=(--tls)' "$script_dir/native-vast.sh" || {
  echo 'FAIL: native controller must receive --tls.' >&2
  exit 1
}

echo 'Stratum TLS launcher tests: OK'
