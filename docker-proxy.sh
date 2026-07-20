#!/usr/bin/env bash
set -euo pipefail

proxy=""
no_proxy="localhost,127.0.0.1,::1"

usage() {
  cat <<'EOF'
Usage:
  bash docker-proxy.sh --proxy http://HOST:PORT [--no-proxy HOSTS]

Configures the Docker daemon proxy used by docker pull. This is separate from
shell HTTP_PROXY variables, which do not affect the Docker daemon.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --proxy) proxy="${2:-}"; shift 2 ;;
    --no-proxy) no_proxy="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) fail "Unknown option: $1" ;;
  esac
done

[[ "$proxy" =~ ^https?://[^[:space:]\"\\]+$ ]] || fail "--proxy must be an http:// or https:// URL without spaces."
[[ "$no_proxy" != *$'\n'* && "$no_proxy" != *$'\r'* && "$no_proxy" != *'"'* && "$no_proxy" != *'\\'* ]] || fail "--no-proxy contains unsupported characters."
command -v systemctl >/dev/null 2>&1 || fail "This host has no systemd Docker service. Configure the Docker daemon proxy through the provider instead."
command -v docker >/dev/null 2>&1 || fail "Docker is not installed."

if ((EUID == 0)); then
  as_root=()
elif command -v sudo >/dev/null 2>&1; then
  as_root=(sudo)
else
  fail "Run as root or install sudo."
fi

proxy_dir="/etc/systemd/system/docker.service.d"
proxy_file="$proxy_dir/http-proxy.conf"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

printf '%s\n' \
  '[Service]' \
  "Environment=\"HTTP_PROXY=$proxy\"" \
  "Environment=\"HTTPS_PROXY=$proxy\"" \
  "Environment=\"NO_PROXY=$no_proxy\"" \
  > "$tmp_file"

"${as_root[@]}" install -d -m 0755 "$proxy_dir"
"${as_root[@]}" install -m 0600 "$tmp_file" "$proxy_file"
"${as_root[@]}" systemctl daemon-reload
echo "Restarting Docker so image pulls use the configured proxy..."
"${as_root[@]}" systemctl restart docker
docker info >/dev/null
echo "Docker daemon proxy configured. Retry with: bash start.sh"
