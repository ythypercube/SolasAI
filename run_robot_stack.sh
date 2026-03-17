#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR="$SCRIPT_DIR/model"
LOG_DIR="$SCRIPT_DIR/.run-logs"
mkdir -p "$LOG_DIR"

read_env_value() {
  local key=$1
  local default_value=$2
  local env_file="$SCRIPT_DIR/.env"
  if [[ ! -f "$env_file" ]]; then
    printf '%s' "$default_value"
    return
  fi

  local line
  line=$(grep -E "^${key}=" "$env_file" | tail -n 1 || true)
  if [[ -z "$line" ]]; then
    printf '%s' "$default_value"
    return
  fi

  local value=${line#*=}
  value=${value%$'\r'}
  printf '%s' "$value"
}

DISPLAY_PORT=$(read_env_value "PORT" "8787")
DISPLAY_ROBOT_PORT=$(read_env_value "ROBOT_BRIDGE_PORT" "8900")

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

NODE_CMD="node"
if ! command -v "$NODE_CMD" >/dev/null 2>&1; then
  echo "Error: Node.js is not installed or not on PATH." >&2
  exit 1
fi

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Error: Python is not available: $PYTHON_CMD" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/inference_server.py" ]]; then
  echo "Error: Missing $MODEL_DIR/inference_server.py" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/server.js" ]]; then
  echo "Error: Missing $SCRIPT_DIR/server.js" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/robot_bridge.py" ]]; then
  echo "Error: Missing $SCRIPT_DIR/robot_bridge.py" >&2
  exit 1
fi

PIDS=()
NAMES=()

cleanup() {
  local exit_code=$?
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "Stopping SolasAI processes..."
    for pid in "${PIDS[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

start_process() {
  local name=$1
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "Starting $name ..."
  (
    cd "$SCRIPT_DIR"
    "$@"
  ) >"$log_file" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  NAMES+=("$name")
  echo "  PID $pid  log: $log_file"
}

start_process inference "$PYTHON_CMD" "$MODEL_DIR/inference_server.py"
start_process backend "$NODE_CMD" "$SCRIPT_DIR/server.js"
start_process robot_bridge "$PYTHON_CMD" "$SCRIPT_DIR/robot_bridge.py"

echo
echo "SolasAI robot stack is starting."
echo "Backend:        http://127.0.0.1:${DISPLAY_PORT}"
echo "Inference:      http://127.0.0.1:8788"
echo "Robot bridge:   http://127.0.0.1:${DISPLAY_ROBOT_PORT}"
echo "Logs folder:    $LOG_DIR"
echo
echo "Press Ctrl+C to stop everything."
echo

while true; do
  for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    name=${NAMES[$i]}
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited. Check $LOG_DIR/${name}.log" >&2
      exit 1
    fi
  done
  sleep 2
done
