"""
SolasGPT training script.
Trains a small transformer LM on conversations.txt.
Saves model checkpoint + vocab to model_checkpoint.pt.

Usage:
  python train.py
  python train.py --epochs 200 --data data/conversations.txt
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from model import SolasGPT

# ─────────────────── config ────────────────────
BLOCK_SIZE = 256
BATCH_SIZE = 16
EMBED_DIM  = 192
NUM_HEADS  = 6
NUM_LAYERS = 6
DROPOUT    = 0.1
LR         = 2e-4
EVAL_EVERY = 50
CHECKPOINT = os.path.join(os.path.dirname(__file__), "model_checkpoint.pt")
# ────────────────────────────────────────────────


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text if c in stoi]


def decode(ids: list[int], itos: dict) -> str:
    return ''.join(itos[i] for i in ids)


def get_batches(data: torch.Tensor, batch_size: int, block_size: int, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, device, iters=50):
    model.eval()
    losses = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        total = 0.0
        for _ in range(iters):
            x, y = get_batches(data, batch_size, block_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / iters
    model.train()
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--data', type=str, default='data/conversations.txt')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, args.data)

    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}  |  Data chars: {len(text)}")

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SolasGPT(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    ).to(device)

    print(f"Model parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    start = time.time()

    for step in range(1, args.epochs + 1):
        x, y = get_batches(train_data, args.batch, BLOCK_SIZE, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_EVERY == 0 or step == args.epochs:
            stats = estimate_loss(model, train_data, val_data, args.batch, BLOCK_SIZE, device)
            elapsed = time.time() - start
            print(f"Step {step:>5} | train={stats['train']:.4f} val={stats['val']:.4f} "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

            if stats['val'] < best_val_loss:
                best_val_loss = stats['val']
                torch.save({
                    'model_state': model.state_dict(),
                    'stoi': stoi,
                    'itos': itos,
                    'config': {
                        'vocab_size': vocab_size,
                        'embed_dim': EMBED_DIM,
                        'num_heads': NUM_HEADS,
                        'num_layers': NUM_LAYERS,
                        'block_size': BLOCK_SIZE,
                        'dropout': DROPOUT,
                    }
                }, CHECKPOINT)
                print(f"  ✓ Checkpoint saved (val={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT}")

    # Quick generation test
    print("\n--- Sample generation ---")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    prompt = "User: hello\nAssistant:"
    context = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=80, temperature=0.8, top_k=40)
    print(decode(out[0].tolist(), itos))


if __name__ == '__main__':
    main()
