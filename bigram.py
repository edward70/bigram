import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
block_size = 8
learning_rate = 1e-2
eval_interval = 300
max_iters = 3000
n_embd = 32

torch.manual_seed(1337)

# read training dataset
with open('input.txt', encoding='utf-8') as f:
  text = f.read()
  print(text[0:100])

chars = sorted(set(text)) # get list of characters
vocab_size = len(chars)

# convert characters to integer tokens
atoi = {c: i for i, c in enumerate(chars)}
itoa = {i: c for i, c in enumerate(chars)}
encode = lambda string: [atoi[c] for c in string]
decode = lambda lst: "".join([itoa[item] for item in lst])

# convert data to tensor
tt = torch.tensor(encode(text), dtype=torch.long)

# train test split
n = int(0.9 * len(tt))
train = tt[:n]
val = tt[n:]

# batching
def get_batch(split):
  data = train if split == 'train' else val
  ix = torch.randint(len(data) - block_size - 1, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
  return x.to(device), y.to(device)

xb, yb = get_batch('train')

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
  
  def forward(self, idx, targets=None):
    tok_embd = self.token_embedding_table(idx) # (B,T,C)
    logits = self.lm_head(tok_embd) # (B, T, vocab_size)
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      # use only the latest from the time dimension
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      samples = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, samples), dim=1)
    return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = mm(X, Y)
            losses[k] = torch.item()
        out[split] = losses.mean()
    model.train()
    return out

mm = BigramLanguageModel()
mm.to(device)

# create a Pytorch optimizer
optimizer = torch.optim.AdamW(mm.parameters(), lr=1e-3)

for steps in range(10000):
  if steps % eval_interval == 0:
    losses = estimate_loss()
    print(step, losses)

  xb, yb = get_batch('train')
  logits, loss = mm(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
print(loss.item())

print(decode(mm.generate(torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))
