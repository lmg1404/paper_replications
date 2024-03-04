import torch 
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

# hyper parameter
D_PROB = 0.1

class Head(nn.Module):
    """Just one head of multihead attention"""
    def __init__(self, head_dim: int, embed_dim: int, mask: bool):
        super().__init__()
        self.key = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.query = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.value= nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(head_dim, head_dim), diagonal=1))
        self.mask_bool = mask
        
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, key, query, value):
        # t for tokens, so I think this is words in this case
        _, t, d_k = key.size() 
        k = self.key(key)
        q = self.query(query)
        v = self.value(value)
        qk = (q@k.transpose(-1, -2)) / d_k ** 0.5
        
        if self.mask_bool:
            qk = qk.masked_fill(self.mask[:t, :t] == 0, float('-inf'))
            
        # 0 is among batches so ith, jth inputs in each batch add to 1, we don't want this
        # 1 is through the columns, which is the word so maybe
        # 2 is the word vector entirely
        qk = F.softmax(qk, dim=-1)
        attn = qk @ v
        
        # dropout occurs at the end of the sublayer before adding residual connections and layernorming
        attn = self.dropout(attn)
        
        return attn 
        
class MultiHeadAttention(nn.Module):
    """Multiply above head by h, which is the number of heads. Key that this is a sublayer"""
    def __init__(self, num_heads: int, head_dim: int, embed_dim: int, mask: bool):
        super().__init__()
        self.multihead = [Head(head_dim, embed_dim, mask) for _ in range(num_heads)]
        # TODO: figure out dims
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    
    def forward(self, key, query, value):
        # we know dim -1 will be dim_model // num_heads so cat returns to dim_model
        cat = torch.cat([head(key, query, value) for head in self.multihead], dim=-1)
        out = self.linear(cat)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        # see section 3.3
        self.sublayer = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=4 * embed_dim),
            nn.ReLU(),
            nn.Linear(in_features=4 * embed_dim, out_features=embed_dim)
        )
    
    def forward(self, x):
        return self.sublayer(x)

class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.mh = MultiHeadAttention(num_heads, head_dim, embed_dim, False)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim) 
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, x):
        x = self.mh(x, x, x)
        x = self.ln1(x + self.dropout(x))
        x = self.ff(x)
        x = self.ln2(x + self.dropout(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, embed_dim: int):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.mh1 = MultiHeadAttention(num_heads, head_dim, embed_dim, True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mh2 = MultiHeadAttention(num_heads, head_dim, embed_dim, False)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff  = FeedForward(embed_dim) # TODO: fix
        self.ln3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, x, enc: torch.Tensor):
        assert x.size() == enc.size(), f"Encoder output and decoder sublayer 1 output must be same shape: {enc.size()} {x.size()}"
        x = self.mh1(x, x, x)
        x = self.ln1(x + self.dropout(x))
        x = self.mh2(enc, x, enc)
        x = self.ln2(x + self.dropout(x))
        x = self.ff(x)
        x = self.ln3(x + self.dropout(x))
        return x

# TODO
class Transformer(nn.Module):
    def __init__(self, N: int, num_heads: int, embed_dim: int, vocab_size: int, context: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim) # (vocabsize x embed_dim) each token/word will have a 512 dim representation
        self.out_emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(*[EncoderBlock(num_heads, embed_dim) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderBlock(num_heads, embed_dim) for _ in range(N)])
        self.lin = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(D_PROB)
        self.register_buffer('pos_enc', self.pos_encoding(context, embed_dim))
    
    def forward(self, src, trg):
        x = self.decode(trg, self.encode(src))
        x = self.lin(x)
        return x
    
    # better to separate encode and decode so we can see what's happening downstream during inference
    def encode(self, x):
        x = self.in_emb(x)
        x = self.dropout(x + self.pos_enc)
        x = self.encoder(x)
        return x
    
    def decode(self, x, src):
        x = self.out_emb(x)
        x = self.dropout(x + self.pos_enc)
        for decoder in self.decoder:
            x = decoder(x, src)
        return x
    
    def pos_encoding(self, max_len, d_model):
        encoding = torch.zeros(max_len, d_model) # don't care about batches broadcasting will fix this
        pow = 2 * torch.arange(0, d_model//2) / d_model
        denom = 10_000 ** pow
        pos = torch.arange(0, max_len).view(-1, 1)
        encoding[:, ::2] = torch.sin(pos/denom) # evens
        encoding[:, 1::2] = torch.cos(pos/denom) # odds
        return encoding
    
    # TODO: inference
    def generate(self):
        pass
    
class DataHolder(Dataset):
    def __init__(self):
        super().__init__()