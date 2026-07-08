# SolasAI Unified Core - Configuration

## Project Structure

```
ai-core/
├── datasets/           # Specialized training datasets
│   ├── conversation/  # General chat & dialogue
│   ├── coding/        # Programming Q&A & examples
│   └── minecraft/     # Game mechanics & behaviors
├── models/            # Model architecture & checkpoints
├── training/          # Training scripts & utilities
└── inference/         # Inference servers & APIs
```

## Dataset Descriptions

### 1. Conversation Dataset
**Purpose**: Train the AI for natural dialogue, greetings, general knowledge
**Files**:
- `general_chat.txt` - Base conversation training data
- `expanded_dialogue.txt` - Extended conversational corpus
- `chat_history.jsonl` - Real user interaction logs

### 2. Coding Dataset
**Purpose**: Train the AI for programming help, debugging, code generation
**Files**:
- `code_examples.txt` - Programming Q&A examples
- `autofix_examples.txt` - Bug fixes and corrections

### 3. Minecraft Dataset
**Purpose**: Train the AI for Minecraft gameplay, strategies, commands
**Files**:
- `gameplay_knowledge.txt` - Minecraft mechanics and strategies
- `youtube_tutorials.txt` - Extracted lesson content
- `player_behavior.json` - Behavioral patterns

## Training Modes

You can train the model on different dataset combinations:

1. **Full Model** (all datasets): Best overall performance
2. **Conversation-only**: Focused on chat/dialogue
3. **Minecraft specialist**: Optimized for game assistance
4. **Code assistant**: Programming helper mode

## Usage

### Training
```bash
cd ai-core/training
python train_model.py --dataset conversation  # Train on conversation data
python train_model.py --dataset minecraft     # Train on Minecraft data
python train_model.py --dataset coding        # Train on coding data
python train_model.py --dataset all           # Train on combined datasets
```

### Inference
```bash
cd ai-core/inference
python chat_server.py --port 8788             # Start chat API server
```

### Model Location
Trained checkpoints are saved in: `ai-core/models/model_checkpoint.pt`
