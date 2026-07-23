#!/usr/bin/env bash
# Self-repair and update helper for copied TensorCash launcher packages.
# Runtime state is ignored by Git and intentionally survives every repair.

readonly TENSORCASH_LAUNCHER_DEFAULT_REPO='https://github.com/Avalonbtc/tensorcash-miner-launcher.git'

launcher_auto_update_enabled() {
  local config_path="${1:-}" value="${TENSORCASH_AUTO_UPDATE:-}"
  if [[ -z "$value" && -n "$config_path" && -r "$config_path" ]]; then
    value="$(sed -n -E 's/^[[:space:]]*TENSORCASH_AUTO_UPDATE[[:space:]]*=[[:space:]]*//p' "$config_path" | tail -n 1)"
    value="${value%\"}"
    value="${value#\"}"
  fi
  case "${value:-true}" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    *)
      echo "TENSORCASH_AUTO_UPDATE must be true or false." >&2
      return 2
      ;;
  esac
}

launcher_command_starts_runtime() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --stop|--status|--logs|--plan|--purge-runtime) return 1 ;;
    esac
  done
  return 0
}

launcher_repo_is_valid() {
  local root="$1"
  [[ -d "$root/.git" ]] || return 1
  [[ "$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true)" == "$root" ]]
}

launcher_repair_git_metadata() {
  local root="$1" repo_url="$2" branch="$3" parent temporary
  parent="$(dirname "$root")"
  temporary="$(mktemp -d "$parent/.tensorcash-launcher-repair.XXXXXX")" || return 1
  echo "Repairing TensorCash launcher Git metadata from ${repo_url}..." >&2
  if ! git clone --depth=1 --branch "$branch" "$repo_url" "$temporary/repo"; then
    rm -rf "$temporary"
    return 1
  fi
  rm -rf "$root/.git"
  mv "$temporary/repo/.git" "$root/.git"
  rmdir "$temporary/repo" "$temporary"
}

launcher_sync_latest() {
  local root="$1"
  local repo_url="${TENSORCASH_LAUNCHER_REPO_URL:-$TENSORCASH_LAUNCHER_DEFAULT_REPO}"
  local branch="${TENSORCASH_LAUNCHER_BRANCH:-main}"
  [[ -d "$root" ]] || { echo "Launcher root is missing: $root" >&2; return 1; }
  command -v git >/dev/null 2>&1 || { echo "Git is required for forced launcher updates." >&2; return 1; }
  [[ "$branch" =~ ^[A-Za-z0-9._/-]+$ ]] || { echo "Invalid TENSORCASH_LAUNCHER_BRANCH." >&2; return 1; }
  [[ "$repo_url" =~ ^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git$ ]] || {
    echo "TENSORCASH_LAUNCHER_REPO_URL must be a GitHub HTTPS repository URL." >&2
    return 1
  }

  if ! launcher_repo_is_valid "$root"; then
    launcher_repair_git_metadata "$root" "$repo_url" "$branch" || {
      echo "Could not repair TensorCash launcher Git metadata." >&2
      return 1
    }
  else
    git -C "$root" remote set-url origin "$repo_url"
  fi

  echo "Synchronizing TensorCash launcher to origin/${branch}..." >&2
  git -C "$root" fetch --prune origin "+refs/heads/${branch}:refs/remotes/origin/${branch}" || return 1
  # This repository intentionally owns only launcher scripts. Runtime data,
  # model caches, logs, and miner.env are ignored and survive both commands.
  git -C "$root" reset --hard "refs/remotes/origin/${branch}" || return 1
  git -C "$root" clean -fd || return 1
  printf 'TensorCash launcher synchronized: %s\n' "$(git -C "$root" rev-parse --short HEAD)" >&2
}
