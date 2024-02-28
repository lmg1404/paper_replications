import torch 
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

# hyper parameters
# subject to change
VOCAB_SIZE = 3  # hyperparameter depending on the task
HEAD_SIZE = 8
EMBED_DIM = 512
HEAD_DIM = EMBED_DIM // HEAD_SIZE
D_PROB = 0.1

class Head(nn.Module):
    """Just one head of multihead attention"""
    def __init__(self, head_dim: int, embed_dim: int, mask: bool):
        super().__init__()
        self.key = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.query = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.value= nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        
        # TODO: figure out register buffer
        # TODO: prototype tril
        self.register_buffer('mask', torch.tril(torch.ones(head_dim, head_dim), diagonal=1))
        self.mask_bool = mask
        
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, x):
        # t for tokens, so I think this is words in this case
        _, t, d_k = x.size() 
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
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
    
    def forward(self, x):
        # we know dim -1 will be dim_model // num_heads so cat returns to dim_model
        cat = torch.cat([head(x) for head in self.multihead], dim=-1)
        out = self.linear(cat)
        return out

# TODO: prototype to finish
class FeedForward(nn.Module):
    def __init__(self):
        self.sublayer = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Linear()
        )
    
    def forward(self, x):
        return self.sublayer(x)

# TODO
class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
         self.mh = MultiHeadAttention(num_heads, head_dim)
         self.ln1 = nn.LayerNorm()
         self.ff = FeedForward() # TODO: change once prototyped
         self.ln2 = nn.LayerNorm()
    
    def forward(self, x):
        x = self.ln1(x + self.multihead(x))
        x = self.ln2(x + self.ff(x))
        return x

# TODO
class DecoderBlock(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        self.mh1 = MultiHeadAttention(num_heads, head_dim)
        self.ln1 = nn.LayerNorm()
        self.mh2 = MultiHeadAttention(num_heads, head_dim)
        self.ln2 = nn.LayerNorm()
        self.ff  = FeedForward() # TODO: fix
        self.ln3 = nn.LayerNorm()
    
    def forward(self, x, enc: torch.Tensor):
        x = self.ln1(x + self.mh1(x))
        x = self.ln2(x + self.mh2(x + enc)) # TODO: fix
        x = self.ln3(x + self.ff(x))
        return x

# TODO
class Transformer(nn.Module):
    def __init__(self, N: int, num_heads: int, head_dim: int):
        self.in_emb = nn.Embedding()
        self.out_emb = nn.Embedding()
        self.encoder = nn.Sequential(*[EncoderBlock(num_heads, head_dim) for _ in range(N)])
        self.decoder = nn.Sequential(*[DecoderBlock(num_heads, head_dim) for _ in range(N)])
        self.lin = nn.Linear() # TODO: fix
        self.sm = nn.Softmax()
    
    def forward(self, x):
        x_enc = self.in_emb(x)
        x_dec = self.out_emb(x)
        
        # TODO: additive positional encodings here
        
        x_enc = self.encoder(x_enc)
        x_dec = self.decoder(x_dec, x_enc)
        x_dec = self.lin(x_dec)
        x_dec = self.sm(x_dec)
        return x_dec