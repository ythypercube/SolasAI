# Dataset Overview - SolasAI Training Data

## Dataset Architecture

SolasAI uses a **three-pillar approach** where one unified AI model can be trained on specialized datasets:

```
┌─────────────────────────────────────────────┐
│         SolasGPT Transformer Model          │
│    (One architecture, multiple training)    │
└─────────────────────────────────────────────┘
              ▲         ▲         ▲
              │         │         │
    ┌─────────┼─────────┼─────────┼─────────┐
    │         │         │         │         │
┌───▼───┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐   │
│ Conv. │ │ Coding│ │Minecr.│ │  All  │   │
│Dataset│ │Dataset│ │Dataset│ │Combine│   │
└───────┘ └───────┘ └───────┘ └───────┘   │
    │         │         │         │         │
    └─────────────────────────────────────────┘
```

## 1. Conversation Dataset 💬

**Location**: `ai-core/datasets/conversation/`

**Purpose**: Train the AI for natural dialogue, greetings, general knowledge questions

**Files**:
- `general_chat.txt` - Base conversational training data
- `expanded_dialogue.txt` - Extended corpus with varied topics
- `chat_history.jsonl` - Real user interactions (logged sessions)

**Example Training Format**:
```
Q: Hello! How are you?
A: Hi there! I'm doing great, thanks for asking. How can I help you today?

Q: What's your name?
A: I'm SolasAI, your helpful assistant. What can I do for you?

Q: Tell me about yourself
A: I'm an AI trained to assist with conversations, coding, and Minecraft gameplay.
```

**Use Cases**:
- Chatbot applications
- General Q&A assistant
- Friendly conversational AI
- Customer service automation

---

## 2. Coding Dataset 💻

**Location**: `ai-core/datasets/coding/`

**Purpose**: Train the AI to help with programming, debugging, and code generation

**Files**:
- `code_examples.txt` - Programming Q&A with code snippets
- `autofix_examples.txt` - Bug fixes and corrections

**Example Training Format**:
```
Q: How do I read a JSON file in Python?
A: Use the json module:
```python
import json

with open('data.json', 'r') as f:
    data = json.load(f)
    print(data)
```

Q: What's wrong with this code: if x = 5:
A: You're using assignment (=) instead of comparison (==). Should be: if x == 5:
```

**Supported Languages**:
- Python
- JavaScript/Node.js
- Java
- Shell/Bash
- SQL
- HTML/CSS

**Use Cases**:
- Programming assistant
- Code review helper
- Debugging support
- Learning tool for beginners

---

## 3. Minecraft Dataset ⛏️

**Location**: `ai-core/datasets/minecraft/`

**Purpose**: Train the AI for Minecraft gameplay strategies, mechanics, and bot behavior

**Files**:
- `gameplay_knowledge.txt` - Game mechanics, strategies, recipes
- `youtube_tutorials.txt` - Extracted lesson content from videos
- `player_behavior.json` - Behavioral patterns and decision logs

**Example Training Format**:
```
Q: How do I find diamonds in Minecraft?
A: Mine at Y-level -59 to -64 in version 1.18+. Use iron pickaxe or better. Strip mining is most efficient. Look for exposed diamonds in caves.

Q: Best strategy for fighting Enderman?
A: Stand under 2-block tall ceiling so Enderman can't teleport to you. Use water bucket to damage them. Don't look directly at their face.

Q: How to make a Nether portal?
A: Build 4x5 obsidian frame (minimum). Use flint and steel to ignite. You need 10 obsidian blocks minimum for the frame.
```

**Knowledge Areas**:
- Combat tactics
- Resource gathering
- Crafting recipes
- Survival strategies
- Redstone mechanics
- Building techniques
- PvP strategies

**Use Cases**:
- Minecraft bot AI (autonomous gameplay)
- Player helper/tutorial system
- Strategy guide
- Command interpretation

---

## 4. Combined Dataset (All) 🌐

**Location**: Virtual (combines all three)

**Purpose**: Train a general-purpose AI with knowledge across all domains

**Training Command**:
```bash
./train-ai.sh all
```

**Characteristics**:
- Largest training corpus
- Most versatile responses
- Can handle mixed queries
- Best for production deployments

**Example Mixed Query Handling**:
```
User: "Write Python code to calculate Minecraft diamond mining efficiency"

Response: "Here's a Python script:
```python
def mining_efficiency(time_minutes, diamonds_found):
    diamonds_per_hour = (diamonds_found / time_minutes) * 60
    print(f'Efficiency: {diamonds_per_hour:.2f} diamonds/hour')
    
mining_efficiency(120, 15)  # 2 hours, 15 diamonds found
```
At Y-level -59, you should average 1-2 diamonds per chunk mined."
```

---

## Dataset Statistics

| Dataset      | Files | Approx Size | Training Time (600 epochs) | Best For                |
|--------------|-------|-------------|----------------------------|-------------------------|
| Conversation | 3     | ~50KB       | 30-45 min                  | Chatbots                |
| Coding       | 2     | ~30KB       | 20-30 min                  | Programming assistants  |
| Minecraft    | 3     | ~80KB       | 45-60 min                  | Game bots               |
| **All**      | 8     | ~160KB      | 90-120 min                 | General purpose         |

*Times approximate on modern CPU (no GPU)*

---

## Adding Your Own Data

### Format Guidelines

**Text Format** (`.txt` files):
```
Q: [Question or prompt]
A: [Response with detailed explanation]

Q: [Next question]
A: [Next response]
```

**JSON Format** (`.jsonl` files):
```json
{"user": "question text", "assistant": "response text", "timestamp": "2024-07-06T19:30:00Z"}
{"user": "another question", "assistant": "another response", "timestamp": "2024-07-06T19:31:00Z"}
```

### Best Practices

1. **Be specific**: Include context and detailed explanations
2. **Use examples**: Code snippets, step-by-step instructions
3. **Stay consistent**: Match the tone of existing data
4. **Quality over quantity**: 100 good examples > 1000 poor ones
5. **Test incrementally**: Train with small epochs first to verify

### Adding New Files

1. Create your dataset file in the appropriate folder:
   - `ai-core/datasets/conversation/my_data.txt`
   - `ai-core/datasets/coding/my_examples.txt`
   - `ai-core/datasets/minecraft/my_strategies.txt`

2. The training script automatically loads all `.txt` files in each directory

3. Train the model:
   ```bash
   ./train-ai.sh [dataset-type]
   ```

---

## Quality Validation

### Signs of Good Training
- Loss decreasing steadily
- Validation loss staying close to training loss
- Generated text is coherent and on-topic

### Signs of Problems
- Validation loss increasing (overfitting)
- Repetitive or nonsensical output
- Training loss stuck high (underfitting)

### Solutions
- **Overfitting**: Reduce epochs, add more varied data
- **Underfitting**: Increase model size, train longer
- **Poor quality**: Review and improve training data

---

## Dataset Roadmap

### Planned Additions
- [ ] More coding languages (Rust, Go, TypeScript)
- [ ] Advanced Minecraft mods/mechanics
- [ ] Multi-turn conversation context
- [ ] Domain-specific knowledge bases

### Community Contributions
Submit training data improvements via pull request to expand AI capabilities!

---

**Last Updated**: 2024-07-06
**Maintainer**: SolasAI Team
