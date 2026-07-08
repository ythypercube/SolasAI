#!/bin/bash
# SolasAI - Train AI Model
# Train the AI on specialized datasets

cd "$(dirname "$0")/SolasGPT/ai-core/training"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  SolasAI Model Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Available datasets:"
echo "  • conversation  - General chat & dialogue"
echo "  • coding        - Programming Q&A"  
echo "  • minecraft     - Game mechanics & strategies"
echo "  • all           - Combined training (recommended)"
echo ""

# Default to conversation if no dataset specified
DATASET="${1:-conversation}"

echo "Training on dataset: $DATASET"
echo ""

python3 train_model.py --dataset "$DATASET" --epochs "${EPOCHS:-600}"
