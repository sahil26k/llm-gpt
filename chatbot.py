import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle

import optuna
import torch.optim as optim
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

batch_size = 64
block_size = 64    
max_iters = 6000   
learning_rate =3e-4
eval_iters = 500    
n_embd = 256      
n_head = 12       
n_layer = 12        
dropout = 0.2   
chars = ""
with open('vocab_wiki_cleaned.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text) | {' '})
vocab_size = (len(chars))
print(len(chars))

string_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_string = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])



class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k= self.key(x)
        q= self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei= wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v=self.value(x)
        out = wei @ v 
        return out
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffws = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        y =self.sa(x)
        x =self.ln1(x+y)
        y =self.ffws(x)
        x = self.ln1(x+y)
        return x
def top_k_top_p_filtering(logits, top_k=50, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., :top_k] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits


class GptLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size , n_embd)
        self.block = nn.Sequential(*[Block(n_embd, n_head = n_head)for _ in range (n_layer)])

        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0 , std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0 , std = 0.02)
        
    
    def forward(self, index, targets= None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device= device))
  
        x= tok_emb + pos_emb #b,t,c
        x = self.block(x)
        x= self.ln_f(x)
        logits = self.lm_head(x)


        
        if targets is None:
            loss = None
        else:
            B, T ,C= logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens ,temperature=1.0, top_k=50, top_p=0.9):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits= logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim =-1)
            index_next  = torch.multinomial(probs, num_samples = 1 )
            index = torch.cat((index,index_next), dim = 1)
        return index
model = GptLanguageModel(vocab_size)
print('loading model')
with open('model-wiki-10.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
print('Model loaded')
m = model.to(device)

while True:
    prompt = input("Enter a prompt:\n")
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(
        context.unsqueeze(0), 
        max_new_tokens=150,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )[0].tolist())
    print(f'Completion:\n{generated_chars}')
    