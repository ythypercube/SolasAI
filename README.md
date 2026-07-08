# SolasAI - Unified Intelligence System

A modular AI system with specialized training datasets for conversation, coding, and Minecraft gameplay.

## 🎯 Quick Start

### Start the Chat Server
```bash
./start-chat-server.sh
```
The AI will be available at `http://localhost:8788`

### Train a Specialized Model
```bash
./train-ai.sh conversation  # Train on dialogue
./train-ai.sh coding        # Train on programming
./train-ai.sh minecraft     # Train on game mechanics
./train-ai.sh all           # Train on everything (recommended)
```

## 📁 Project Structure

```
SolasAI/
├── SolasGPT/                        # Main project folder
│   ├── ai-core/                    # Unified AI system
│   │   ├── datasets/               # Specialized training data
│   │   │   ├── conversation/       # Chat & dialogue
│   │   │   │   ├── general_chat.txt
│   │   │   │   ├── expanded_dialogue.txt
│   │   │   │   └── chat_history.jsonl
│   │   │   ├── coding/             # Programming Q&A
│   │   │   │   ├── code_examples.txt
│   │   │   │   └── autofix_examples.txt
│   │   │   └── minecraft/          # Game knowledge
│   │   │       ├── gameplay_knowledge.txt
│   │   │       ├── youtube_tutorials.txt
│   │   │       └── player_behavior.json
│   │   ├── models/                 # Model architecture
│   │   │   ├── solas_gpt.py       # Transformer model
│   │   │   └── model_checkpoint.pt # Trained weights
│   │   ├── training/               # Training system
│   │   │   └── train_model.py     # Multi-dataset trainer
│   │   ├── inference/              # Inference server
│   │   │   └── chat_server.py     # HTTP API
│   │   ├── requirements.txt        # Python dependencies
│   │   └── README.md               # Detailed docs
│   ├── minecraft-bot-service/      # Minecraft bot integration
│   ├── turbowarp-ai-backend/       # TurboWarp integration
│   ├── fabric-mc-ai-agent/         # Fabric mod
│   ├── MinecraftServer/            # Server files
│   ├── libraries/                  # Java dependencies
│   ├── minecraft_data/             # Game state
│   ├── litematic-build-service/    # Building service
│   └── versions/                   # Minecraft versions
├── start-chat-server.sh            # Launch inference server
├── train-ai.sh                     # Train models
├── run_robot_stack.sh              # Start robot stack
└── README.md                       # This file
```

## 🤖 AI Capabilities

### 1. Conversation Mode
Natural dialogue, general knowledge, friendly chat
- **Dataset**: `SolasGPT/ai-core/datasets/conversation/`
- **Use case**: Chatbot, assistant, general Q&A

### 2. Coding Mode
Programming help, debugging, code examples
- **Dataset**: `SolasGPT/ai-core/datasets/coding/`
- **Use case**: Programming assistant, code generation

### 3. Minecraft Mode
Game mechanics, strategies, bot commands
- **Dataset**: `SolasGPT/ai-core/datasets/minecraft/`
- **Use case**: Minecraft bot AI, gameplay assistant

### 4. Unified Mode (All)
Combined knowledge across all domains
- **Dataset**: All datasets combined
- **Use case**: General-purpose assistant

## 🔧 Configuration

### Environment Variables
```bash
export PORT=8788                    # Server port
export AI_MODE=general              # conversation|coding|minecraft
export USE_EMBEDDINGS=true          # Enable semantic search
export EPOCHS=600                   # Training iterations
```

### Training Configuration
Edit `SolasGPT/ai-core/training/train_model.py`:
- `BLOCK_SIZE`: Context window (default: 256)
- `BATCH_SIZE`: Batch size (default: 16)
- `EMBED_DIM`: Model dimension (default: 192)
- `NUM_HEADS`: Attention heads (default: 6)
- `NUM_LAYERS`: Transformer layers (default: 6)

## 🚀 API Usage

### Chat Endpoint
```bash
curl -X POST http://localhost:8788/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "session_id": "user123"}'
```

### Response
```json
{
  "response": "I'm doing great! How can I help you today?",
  "session_id": "user123"
}
```

## 🎮 Minecraft Bot Integration

The AI powers a Minecraft bot through `minecraft-bot-service/`:
- Autonomous gameplay decisions
- Combat tactics and gear management
- Objective-based behavior (hunt, general1, combat)
- Natural language command interpretation

See [minecraft-bot-service/README.md](minecraft-bot-service/README.md) for details.

## 📊 Dataset Management

### Adding Training Data

**Conversation examples** (`ai-core/datasets/conversation/`):
```
Q: What's the weather like?
A: I don't have access to real-time weather data, but I can chat about weather topics!

Q: Tell me a joke
A: Why did the programmer quit his job? Because he didn't get arrays! 😄
```

**Coding examples** (`ai-core/datasets/coding/`):
```
Q: How do I reverse a string in Python?
A: Use slicing: reversed_str = original[::-1]

Q: Explain list comprehension
A: List comprehension creates lists concisely: [x*2 for x in range(10)]
```

**Minecraft examples** (`ai-core/datasets/minecraft/`):
```
Q: How do I find diamonds?
A: Mine at Y-level -59 to -64 in 1.18+. Use iron pickaxe or better. Strip mine for efficiency.

Q: Best food for healing?
A: Golden carrots for saturation, cooked beef for hunger. Suspicious stew for effects.
```

## 🔄 Migration from Old Structure

The project has been consolidated from scattered folders:
- ~~`/SolasAI/model/`~~ → `ai-core/models/`
- ~~`/model/`~~ → `ai-core/models/`
- ~~`/training_data/`~~ → `ai-core/datasets/`
- ~~`inference_server.py`~~ → `ai-core/inference/chat_server.py`
- ~~`train.py`~~ → `ai-core/training/train_model.py`

Old files preserved for compatibility but should be considered deprecated.

## 🧪 Development

### Test Training
```bash
cd ai-core/training
python train_model.py --dataset conversation --epochs 50
```

### Test Inference
```bash
cd ai-core/inference
python chat_server.py --port 8788
```

### Monitor Training
Watch for validation loss improvements:
```
Step   50 | train=2.1234 val=2.2345 lr=2.0e-04 | 12.5s
Step  100 | train=1.9876 val=2.1234 lr=1.9e-04 | 25.1s
  ✓ Checkpoint saved (val=2.1234)
```

## 📝 License

Open source project for educational and research purposes.

## 🤝 Contributing

1. Add training data to appropriate dataset folder
2. Test with small epoch count first
3. Submit improvements via pull request

---

**Status**: ✅ Unified architecture complete - one AI, three specialized datasets
