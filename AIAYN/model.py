import torch 
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset

# constants
NUM_EMBED = 204
EMBED_DIM = 512
D_PROB = 0.1

# TODO: prototype entire module
class Head(nn.Module):
    """Just one head of multihead attention"""
    def __init__(self, head_dim: int):
        super.__init__()
        self.key = nn.Linear(in_features=NUM_EMBED, out_features=head_dim, bias=False)
        self.query = nn.Linear(in_features=NUM_EMBED, out_features=head_dim, bias=False)
        self.value= nn.Linear(in_features=NUM_EMBED, out_features=head_dim, bias=False)
        
        # TODO: figure out register buffer
        # TODO: prototype tril
        self.register_buffer = ('mask', torch.tril(torch.ones(head_dim, head_dim), diagonal=1))
        
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, x, mask: bool):
        _, _, d_k = x.shape 
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        qk = (q@k.T) / torch.sqrt(d_k)
        
        if mask:
            qk = qk.masked_fill(self.mask, float('-inf'))
        
        qk = nn.Softmax(qk)
        attn = qk @ v
        
        # dropout occurs at the end of the sublayer before adding residual connections and layernorming
        attn = self.dropout(attn)
        
        return attn 
        
# TODO
class MultiHeadAttention(nn.Module):
    """Multiply above head by h, which is the number of heads. Key that this is a sublayer"""
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

# TODO
class FeedForward(nn.Module):
    def __init__(self):
        pass
    
    def forward(self):
        pass

# TODO
class Encoder(nn.Module):
    def __init__(self):
        pass 
    
    def forward(self, x):
        pass

# TODO
class Decoder(nn.Module):
    def __init__(self):
        pass 
    
    def forward(self, x):
        pass

# TODO
class Transformer(nn.Module):
    def __init__(self):
        pass
    
    def forward(self, x):
        pass