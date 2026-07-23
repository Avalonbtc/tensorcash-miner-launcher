#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../launcher-sync.sh
source "$script_dir/../launcher-sync.sh"

TENSORCASH_AUTO_UPDATE=true
launcher_auto_update_enabled
TENSORCASH_AUTO_UPDATE=false
if launcher_auto_update_enabled; then
  echo 'FAIL: disabled launcher update must not be enabled' >&2
  exit 1
fi

config_file="$(mktemp)"
trap 'rm -f "$config_file"' EXIT
printf 'TENSORCASH_AUTO_UPDATE=false\n' > "$config_file"
unset TENSORCASH_AUTO_UPDATE
if launcher_auto_update_enabled "$config_file"; then
  echo 'FAIL: miner.env update setting must be honored' >&2
  exit 1
fi

launcher_command_starts_runtime --pool example:3336
if launcher_command_starts_runtime --status; then
  echo 'FAIL: status must not trigger a launcher update' >&2
  exit 1
fi
if launcher_command_starts_runtime --stop; then
  echo 'FAIL: stop must not trigger a launcher update' >&2
  exit 1
fi

echo 'launcher sync tests: OK'
