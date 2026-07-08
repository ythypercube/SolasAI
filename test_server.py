#!/usr/bin/env python3
"""Quick test server to verify Flask routes work"""
import sys
sys.path.insert(0, 'SolasGPT/ai-core/models')

from flask import Flask, request, jsonify
from solas_gpt import SolasGPT
import torch

app = Flask(__name__)

print("Creating test model...")
model = SolasGPT(vocab_size=256, embed_dim=192, num_heads=6, num_layers=6)
print(f"✓ Model created: {model.param_count():,} parameters")

# Simple character tokenizer for testing
chars = [chr(i) for i in range(256)]
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi.get(c, 0) for c in s[:50]]  # Limit to 50 chars for testing

def decode(indices):
    return ''.join([itos.get(i, '?') for i in indices])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Test server running"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', 'Hello')
    session_id = data.get('session_id', 'test')
    
    # Encode input and generate response
    input_ids = torch.tensor([encode(message)], dtype=torch.long)
    
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=20, temperature=0.8)
    
    response = decode(output[0].tolist())
    
    return jsonify({
        "response": f"Echo (test mode): {message[:30]}...",
        "session_id": session_id,
        "model_used": "test_model",
        "parameters": model.param_count()
    }), 200

@app.route('/chat-plain', methods=['POST'])
def chat_plain():
    message = request.data.decode('utf-8')
    return f"Echo (test mode): {message[:50]}\n", 200

@app.route('/reset', methods=['POST'])
def reset():
    return jsonify({"status": "ok", "message": "Session reset"}), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    return jsonify({"status": "ok", "message": "Feedback received"}), 200

if __name__ == '__main__':
    port = 8788
    print(f"\n{'='*50}")
    print(f"  SolasAI Test Server")
    print(f"{'='*50}")
    print(f"\nServer running on http://localhost:{port}")
    print("\nEndpoints:")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  POST http://localhost:{port}/chat")
    print(f"  POST http://localhost:{port}/chat-plain")
    print(f"  POST http://localhost:{port}/reset")
    print(f"  POST http://localhost:{port}/feedback")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
