#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BACKEND_DIR="$SCRIPT_DIR/SolasGPT/turbowarp-ai-backend"

if [[ ! -f "$BACKEND_DIR/run_robot_stack.sh" ]]; then
  echo "Error: Missing $BACKEND_DIR/run_robot_stack.sh" >&2
  exit 1
fi

echo "Server starting..."
exec bash "$BACKEND_DIR/run_robot_stack.sh"
