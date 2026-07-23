#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher_dir="$(cd "$script_dir/.." && pwd)"
config_file="$(mktemp)"
trap 'rm -f "$config_file" "${config_file}".before-policy-refresh.*' EXIT

cat > "$config_file" <<'EOF'
MINER_IMAGE=ghcr.io/avalonbtc/tensorcash-miner:mainnet-0.1.0
POOL_HOST=stratum.example.com
POOL_PORT=443
POOL_TLS=true
POOL_TLS_INSECURE=false
PAYOUT_ACCOUNT=tc1qexample
WORKER=rig-01
NOMP_SIDECAR_TOKEN=0123456789abcdef0123456789abcdef
MODELS_DATA=/srv/models
RUNTIME_DATA=/srv/runtime
TENSORCASH_HTTP_PROXY=http://127.0.0.1:7890
TENSORCASH_IMAGE_ARCHIVE_URL=https://mirror.example/image.tar.zst
GPU_GROUPS=0,1
TENSORCASH_MODEL_PRECISION=bf16
GPU_MEM_UTIL=0.78
TENSORCASH_CONCURRENCY_MODE=manual
VLLM_MAX_NUM_SEQS=1
NOMP_SIDECAR_CONCURRENCY=1
NOMP_SIDECAR_PREFETCH_REQUESTS=0
CUSTOM_OPERATOR_SETTING=preserve-me
EOF

MINER_CONFIG="$config_file" TENSORCASH_AUTO_UPDATE=false \
  bash "$launcher_dir/start.sh" --refresh-env >/dev/null

assert_line() {
  local expected="$1"
  grep -Fx -- "$expected" "$config_file" >/dev/null || {
    echo "FAIL: missing line: $expected" >&2
    exit 1
  }
}

assert_absent() {
  local expected="$1"
  if grep -Fx -- "$expected" "$config_file" >/dev/null; then
    echo "FAIL: stale policy line remains: $expected" >&2
    exit 1
  fi
}

assert_line 'PAYOUT_ACCOUNT=tc1qexample'
assert_line 'POOL_HOST=stratum.example.com'
assert_line 'POOL_TLS=true'
assert_line 'NOMP_SIDECAR_TOKEN=0123456789abcdef0123456789abcdef'
assert_line 'MODELS_DATA=/srv/models'
assert_line 'TENSORCASH_HTTP_PROXY=http://127.0.0.1:7890'
assert_line 'CUSTOM_OPERATOR_SETTING=preserve-me'
assert_line 'MINER_ENV_POLICY_SCHEMA=2'
assert_line 'GPU_GROUPS=auto'
assert_line 'TENSORCASH_MODEL_PRECISION=auto'
assert_line 'GPU_MEM_UTIL=0.89'
assert_line 'TENSORCASH_CONCURRENCY_MODE=auto'
assert_line 'TENSORCASH_AUTO_CONCURRENCY_START=32'
assert_line 'TENSORCASH_AUTO_CONCURRENCY_CEILING=1024'
assert_absent 'GPU_GROUPS=0,1'
assert_absent 'TENSORCASH_MODEL_PRECISION=bf16'
assert_absent 'VLLM_MAX_NUM_SEQS=1'
assert_absent 'NOMP_SIDECAR_CONCURRENCY=1'

# The public schema variable is deliberately sourceable. Regressions here
# abort every normal launch before Docker/Compose is even reached.
bash -c 'set -euo pipefail; readonly TENSORCASH_MINER_ENV_POLICY_SCHEMA=2; source "$1"; test "$MINER_ENV_POLICY_SCHEMA" = 2' _ "$config_file"

backup_count="$(compgen -G "${config_file}.before-policy-refresh.*" | wc -l | tr -d '[:space:]')"
[[ "$backup_count" == 1 ]] || {
  echo "FAIL: refresh must create one backup, got $backup_count" >&2
  exit 1
}

echo 'miner.env refresh tests: OK'
