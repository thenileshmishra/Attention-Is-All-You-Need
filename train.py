"""
Train a tiny Transformer on a reverse-sequence task.
Run:  python train.py
After training, it prints a few inference examples.
"""

import random
import torch
import torch.nn as nn
from model import Transformer

# ── Hyperparameters (all in one place) ──────────────────────────────
VOCAB_SIZE = 15        # 0 = PAD, 1 = BOS, 2 = EOS, 3..14 = a..l
BOS, EOS, PAD = 1, 2, 0
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
EPOCHS = 15
LR = 3e-4
BATCH = 64
SEQ_LEN = (3, 8)      # min/max source length
N_SAMPLES = 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# simple id ↔ token maps
TOKENS = ["<PAD>", "<BOS>", "<EOS>"] + [chr(ord("a") + i) for i in range(VOCAB_SIZE - 3)]
id2tok = {i: t for i, t in enumerate(TOKENS)}
tok2id = {t: i for i, t in id2tok.items()}


# ── Data generation ─────────────────────────────────────────────────
def make_batch():
    """Generate the entire dataset as two tensors: src (B, L) and tgt (B, L+2)."""
    random.seed(42)
    data_ids = list(range(3, VOCAB_SIZE))
    srcs, tgts = [], []
    for _ in range(N_SAMPLES):
        length = random.randint(*SEQ_LEN)
        s = [random.choice(data_ids) for _ in range(length)]
        t = [BOS] + s[::-1] + [EOS]
        srcs.append(s)
        tgts.append(t)

    # pad to max length in each list
    def pad(seqs):
        ml = max(len(s) for s in seqs)
        return torch.tensor([s + [PAD] * (ml - len(s)) for s in seqs])

    return pad(srcs), pad(tgts)


# ── Training loop ───────────────────────────────────────────────────
def train():
    src_all, tgt_all = make_batch()
    model = Transformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(src_all))
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(src_all), BATCH):
            idx = perm[i : i + BATCH]
            src = src_all[idx].to(DEVICE)
            tgt = tgt_all[idx].to(DEVICE)

            logits = model(src, tgt[:, :-1])                     # teacher forcing
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch:2d}  loss {total_loss / n_batches:.4f}")

    return model


# ── Greedy inference demo ───────────────────────────────────────────
@torch.no_grad()
def demo(model):
    model.eval()
    samples = ["a b c d", "f e a", "k j b a h"]
    for text in samples:
        src_ids = [tok2id[t] for t in text.split()]
        src = torch.tensor([src_ids], device=DEVICE)
        memory = model.encode(src)
        gen = [BOS]
        for _ in range(20):
            tgt = torch.tensor([gen], device=DEVICE)
            logits = model.decode(tgt, memory)
            nxt = logits[0, -1].argmax().item()
            gen.append(nxt)
            if nxt == EOS:
                break
        pred = " ".join(id2tok[i] for i in gen[1:])
        expected = " ".join(reversed(text.split())) + " <EOS>"
        print(f"  src: {text}  |  expected: {expected}  |  got: {pred}")


if __name__ == "__main__":
    model = train()
    print("\n── inference ──")
    demo(model)
