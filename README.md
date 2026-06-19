# NanoGPT — LLM From Scratch
> Built following Andrej Karpathy's NanoGPT series. Character-level transformer trained on Wikipedia text.

---

## Project Structure

```
LLM-From-Scratch/
├── data/
│   └── crawler.py              # Wikipedia scraper
├── data_pipeline/
│   ├── tokenization.py         # Char-level encode/decode
│   └── data_handling.py        # Train/val split + get_batch
├── training/
│   └── Bigram_model.py         # Full transformer model
└── README.md
```

Run via:
```bash
python -m training.Bigram_model
```

---

## Pipeline Overview

```
Wikipedia URLs
     ↓
  Crawler (collects N links, dedupes via JSON, scrapes text)
     ↓
  Tokenization (char → id mapping, encode/decode)
     ↓
  Data Handling (train/val split → get_batch → (B,T) tensors)
     ↓
  BigramLanguageModel (transformer forward pass)
     ↓
  Loss (cross_entropy) → AdamW optimizer
```

---

## Hyperparameters

| Parameter | Value |
|---|---|
| `batch_size` (B) | 64 |
| `block_size` (T) | 256 |
| `n_embd` (C) | 384 |
| `n_heads` | 6 |
| `n_layers` | 6 |
| `dropout` | 0.2 |
| `lr` | 3e-4 |
| `max_iterations` | 10000 |

---

## Data Pipeline

### Crawler
- Collects Wikipedia article links (N at a time, configurable)
- Stores visited links in a JSON file to avoid re-scraping
- Scrapes raw text from each article

### Tokenization (char-level)
- Builds vocabulary from unique characters in corpus
- `encoding`: maps every char → integer id
- `decoding`: maps ids → chars back
- Result: `idx` tensor of shape `(B, T)` = `(64, 256)`

### Data Handling
- `data_splitting`: splits encoded text into train / val
- `get_batch`: randomly samples a batch → returns `(xb, yb)` each `(B, T)`

---

## Model Architecture

### Tensor dimensions (with concrete numbers)

```
Input idx:                  (B, T)        = (64, 256)
Token embedding table:      (vocab_size, C) = (65, 384)
Position embedding table:   (T, C)        = (256, 384)
x = tok_embd + pos_embd:    (B, T, C)     = (64, 256, 384)
  ↓ through n_layers=6 Blocks
x after blocks:             (64, 256, 384)
  ↓ LayerNorm
x after ln_f:               (64, 256, 384)
  ↓ lm_head (Linear → vocab)
logits:                     (B, T, vocab) = (64, 256, 65)
```

---

### Head (Single Self-Attention Head)

```
head_size = n_embd // n_heads = 384 // 6 = 64

x: (B, T, C) = (64, 256, 384)
Q = x @ W_q  → (B, T, head_size) = (64, 256, 64)
K = x @ W_k  → (64, 256, 64)
V = x @ W_v  → (64, 256, 64)

wei = Q @ K.T * C^-0.5   → (B, T, T) = (64, 256, 256)
wei = masked_fill(tril==0, -inf)   ← causal masking
wei = softmax(wei)                 ← attention weights
wei = dropout(wei)

out = wei @ V             → (B, T, head_size) = (64, 256, 64)
```

**Why `-inf` before softmax?** Future tokens must not attend to past context. `-inf → softmax → 0`, so those positions vanish.

**Why scale by `C^-0.5`?** Prevents dot products from getting too large in high dimensions, which would push softmax into flat/saturated regions.

---

### MultiHeadAttention

```
6 heads running in parallel, each producing (B, T, 64)
concat all heads → (B, T, 384)   ← back to full C
projection Linear(384, 384) → (B, T, 384)
dropout
```

---

### FeedForward

```
Linear(384 → 1536)   ← expand 4x
ReLU
Linear(1536 → 384)   ← compress back
Dropout
```

Operates **per-token independently** — no cross-token communication here (that's attention's job).

---

### Block (Transformer Block)

```python
x = x + self.sa(self.ln1(x))    # pre-norm self-attention + residual
x = x + self.ffwd(self.ln2(x))  # pre-norm feedforward + residual
```

**Pre-norm (LayerNorm before sublayer)** — Karpathy's implementation diverges from the original "Attention Is All You Need" paper which used post-norm. Pre-norm trains more stably.

**Residual connections** (`x + ...`) allow gradients to flow directly through the network during backprop, enabling training of deep stacks.

---

### BigramLanguageModel (Full Model)

```
token_embedding_table   : nn.Embedding(vocab_size, n_embd)
position_embedding_table: nn.Embedding(block_size, n_embd)
blocks                  : 6× Block(n_embd, n_heads)
ln_f                    : LayerNorm(n_embd)         ← final norm
lm_head                 : Linear(n_embd, vocab_size)
```

#### forward()
```
idx (B,T) → embeddings → blocks → layernorm → lm_head → logits (B,T,vocab)
logits reshaped: (B*T, vocab) = (16384, 65)
targets reshaped: (B*T,)      = (16384,)
loss = cross_entropy(logits, targets)
```

#### generate()
```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]   # crop to context window
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :]         # last time step only
    probs = softmax(logits)
    idx_next = multinomial(probs, 1)  # sample
    idx = cat(idx, idx_next)          # append
```

#### loss_function() — evaluation only
- Runs `eval_iters=200` batches on both train and val
- Averages loss — more stable estimate than single-batch loss
- Wrapped in `@torch.no_grad()` — no gradients needed, saves memory

#### optimizer() — training loop
```
AdamW optimizer
for epoch in range(10000):
    every 10 steps → print train/val loss
    get_batch → forward → zero_grad → backward → step
after training → generate 200 tokens → decode → print
```

---

## Loss & Output

```
x: (64, 256, 384)
  ↓ lm_head
logits: (64, 256, 65)   ← 65 = vocab_size (assume)
  ↓ reshape
logits: (16384, 65)
targets: (16384,)
  ↓ F.cross_entropy
scalar loss
  ↓ AdamW.step()
```

---

## Known Issues Fixed

| Issue | Fix |
|---|---|
| Dropout applied before projection in MHA | Move `dropout` after `proj` |
| `vocab_size = len(set(text))` fragile | Expose `vocab_size` from `Tokenization` class |
| Head comments show `(B,T,C)` for Q/K/V | Correct shape is `(B,T,head_size)` = `(64,256,64)` |
| `k`-counter in `loss_function` | Replace with named tuple iteration |

---

## Version Log

**v0.1** — Current
- Crawler (Wikipedia)
- Char-level tokenization
- Data handler (split + batch)
- BigramLanguageModel with full transformer (6 layers, 6 heads, 384 embd)
