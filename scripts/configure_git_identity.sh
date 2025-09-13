#!/usr/bin/env bash
set -euo pipefail

SCOPE="local"
FILE="GITHUB_IDENTITY.txt"

usage() {
  echo "Usage: $0 [--global] [--file PATH]" >&2
  echo "  --global       Set globally instead of repo-local" >&2
  echo "  --file PATH    Path to identity file (default: GITHUB_IDENTITY.txt)" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --global)
      SCOPE="global"
      shift
      ;;
    --file)
      FILE="${2:-}"
      if [[ -z "$FILE" ]]; then
        echo "Error: --file requires a path" >&2
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$SCOPE" == "local" ]]; then
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not inside a git repository. Use --global or run inside a repo." >&2
    exit 1
  fi
fi

if [[ ! -f "$FILE" ]]; then
  echo "Error: File not found: $FILE" >&2
  exit 1
fi

# Extract values
extract() {
  local key="$1"
  # shellcheck disable=SC2002
  sed -n "s/^${key}[[:space:]]*:[[:space:]]*//Ip" "$FILE" | head -n1 | sed 's/^\s\+//; s/\s\+$//'
}

USERNAME=$(extract "GitHub Username")
EMAIL=$(extract "Email")

if [[ -z "${USERNAME}" ]]; then
  echo "Error: Could not find 'GitHub Username:' line with a value in $FILE" >&2
  exit 1
fi

if [[ -z "${EMAIL}" ]]; then
  echo "Error: Could not find 'Email:' line with a value in $FILE" >&2
  exit 1
fi

# Very loose email sanity check
if [[ ! "$EMAIL" =~ ^[^@\s]+@[^@\s]+\.[^@\s]+$ ]]; then
  echo "Warning: '$EMAIL' doesn't look like a typical email. Continuing anyway..." >&2
fi

if [[ "$SCOPE" == "global" ]]; then
  git config --global user.name "$USERNAME"
  git config --global user.email "$EMAIL"
else
  git config user.name "$USERNAME"
  git config user.email "$EMAIL"
fi

echo "Configured git ${SCOPE} identity:" >&2
echo "  user.name  = $USERNAME" >&2
echo "  user.email = $EMAIL" >&2
