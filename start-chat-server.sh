#!/bin/bash
# SolasAI - Start Chat Server
# Launches the AI inference server for conversational interactions

cd "$(dirname "$0")/SolasGPT/ai-core/inference"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SolasAI Chat Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Starting unified inference server..."
echo "Port: ${PORT:-8788}"
echo "Mode: ${AI_MODE:-general}"
echo ""

python3 chat_server.py --port "${PORT:-8788}" "$@"
