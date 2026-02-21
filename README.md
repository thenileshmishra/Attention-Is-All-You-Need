# Simplified Transformer (Educational)

A minimal encoder-decoder Transformer in **two files**. Trains on a toy reverse-sequence task so you can focus on the architecture, not the plumbing.

## Files

| File | What it contains |
|---|---|
| `model.py` | PositionalEncoding, MultiHeadAttention, EncoderLayer, DecoderLayer, Transformer |
| `train.py` | Data generation, training loop, greedy-decoding demo |

## Run

```bash
pip install torch
python train.py
```

That's it. It trains for ~15 epochs and prints inference examples at the end.

## What you'll learn by reading the code

1. **Positional encoding** — sinusoidal position signal added to embeddings.
2. **Multi-head attention** — Q/K/V projections, scaled dot-product, head splitting.
3. **Encoder layer** — self-attention → residual + LayerNorm → feed-forward → residual + LayerNorm.
4. **Decoder layer** — masked self-attention → cross-attention over encoder output → feed-forward.
5. **Causal mask** — upper-triangular mask that prevents the decoder from looking ahead.
6. **Teacher forcing** — feeding ground-truth target tokens during training.
7. **Greedy decoding** — generating one token at a time at inference.


