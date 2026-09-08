import sys
sys.path.insert(0, 'SolasGPT/ai-core/models')
import torch
from solas_gpt import SolasGPT

# Load checkpoint
checkpoint = torch.load('SolasGPT/ai-core/models/model_checkpoint.pt', map_location='cpu')
model = SolasGPT(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state'])
model.eval()

stoi = checkpoint['stoi']
itos = checkpoint['itos']

def generate(prompt, max_new=150):
    context = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0)
    output = model.generate(context, max_new_tokens=max_new, top_k=40, temperature=0.8)
    return ''.join([itos[i] for i in output[0].tolist()])

print("=== Testing SolasGPT ===\n")
print("Prompt 1: 'hello'")
print(generate("hello", 100))
print("\n" + "="*50 + "\n")

print("Prompt 2: 'How do I write a for loop in Python?'")
print(generate("How do I write a for loop in Python?", 200))
print("\n" + "="*50 + "\n")

print("Prompt 3: 'What is Minecraft?'")
print(generate("What is Minecraft?", 150))
