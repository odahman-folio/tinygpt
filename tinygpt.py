import torch
import torch.nn as nn
import torch.nn.functional as F

with open("$YOUR_TEXT_FILE.txt", encoding="utf-8") as f: #Change $YOUR_TEXT_FILE with the name of your .txt file
    text = f.read()

marker = "$MARKER" #Replace $MARKER with wherever you want the reader to stop tokenising
if marker in text:
    text = text[:text.index(marker)]

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
device = "cuda" if torch.cuda.is_available() else "cpu"

n = int(0.9 * len(data))
train_data = data[:n].to(device)
val_data = data[n:].to(device)

print("total tokens:", len(data))
print("train:", len(train_data), "  val:", len(val_data), "  device:", device)
print("first 20 tokens:", train_data[:20].tolist())

n_embd = 128
batch_size = 64
block_size = 32
num_heads=4
n_layer = 6
dropout = 0.2
TRAIN = True

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size]   for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(100)
        for k in range(100):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

xb, yb = get_batch("train")
print("x shape:", xb.shape, " y shape:", yb.shape)
print("x[0]:", xb[0].tolist())
print("y[0]:", yb[0].tolist())

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                                   # (B, T, head_size)
        q = self.query(x)                                 # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)                                 # (B, T, head_size)
        out = wei @ v                                     # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)              # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)           # (T, n_embd)
        x = tok_emb + pos_emb                            # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

model = GPTModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

if TRAIN:
    for step in range(3000):
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")
    final = estimate_loss()
    torch.save(model.state_dict(), "model.pt")
    print("saved weights to model.pt")
    print(f"final  train {final['train']:.4f}  val {final['val']:.4f}")
else:
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()
    print("loaded weights from model.pt")

prompt = "$YOUR_PROMPT_HERE" #Replace with your seed text. Every character in it must appear somewhere in your training file, or encode() will KeyError
start = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
out = model.generate(start, max_new_tokens=500, temperature=0.7, top_k=40)
print(decode(out[0].tolist()))
