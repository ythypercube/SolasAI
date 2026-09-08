#!/bin/bash
# SolasAI - Start Chat Server
# Launches the AI inference server for conversational interactions

PROJECT_ROOT="$(dirname "$0")"
cd "$PROJECT_ROOT"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SolasAI Chat Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting unified inference server..."
echo "Port: ${PORT:-8788}"
echo "Mode: ${AI_MODE:-general}"
echo ""

.venv/bin/python3 SolasGPT/ai-core/inference/chat_server.py --port "${PORT:-8788}" "$@"
