import torch 
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
import time
from torch.nn.utils.rnn import pad_sequence


# hyper parameter
D_PROB = 0.1

# ---------------------------------------------------------------------------------------------
#                             HELPER CLASSES
# ---------------------------------------------------------------------------------------------

class Head(nn.Module):
    """Just one head of multihead attention"""
    def __init__(self, head_dim: int, embed_dim: int, mask: bool):
        super().__init__()
        self.key = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.query = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)
        self.value = nn.Linear(in_features=embed_dim, out_features=head_dim, bias=False)

        self.register_buffer('mask', torch.tril(torch.ones(head_dim, head_dim), diagonal=1))
        self.mask_bool = mask
        
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, key, query, value, padding_mask=None):
        # t for tokens, so I think this is words in this case
        _, t, d_k = key.size() 
        k = self.key(key)
        q = self.query(query)
        v = self.value(value)
        qk = (q@k.transpose(-2, -1)) / d_k ** 0.5

        if self.mask_bool:
            try:
                qk = qk.masked_fill(self.mask[:t, :t] == 0, float('-inf'))
            except Exception as e:
                assert f"{Exception}, shapes qk: {qk.size()} and mask: {self.mask[:t, :t].size()}"
        # TODO: padding mask
        if torch.is_tensor(padding_mask):
            qk = qk.masked_fill(padding_mask == 1, float('-inf'))
        # 0 is among batches so ith, jth inputs in each batch add to 1, we don't want this
        # 1 is through the columns, which is the word so maybe
        # 2 is the word vector entirely
        qk = F.softmax(qk, dim=-1)
        attn = torch.matmul(qk, v)
        
        # dropout occurs at the end of the sublayer before adding residual connections and layernorming
        attn = self.dropout(attn)
        
        return attn 
        
class MultiHeadAttention(nn.Module):
    """Multiply above head by h, which is the number of heads. Key that this is a sublayer"""
    def __init__(self, num_heads: int, head_dim: int, embed_dim: int, mask: bool):
        super().__init__()
        # this needs to be a module list otherwise it won't be moved to the GPU
        self.multihead = nn.ModuleList([Head(head_dim, embed_dim, mask) for _ in range(num_heads)])
        # TODO: figure out dims
        self.linear = nn.Linear(in_features=embed_dim, out_features=embed_dim)
    
    def forward(self, key, query, value, padding_mask=None):
        # we know dim -1 will be dim_model // num_heads so cat returns to dim_model
        cat = torch.cat([head(key, query, value, padding_mask) for head in self.multihead], dim=-1)
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
    
    def forward(self, x, padding_mask):
        x = self.mh(x, x, x, padding_mask)
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
        self.ff  = FeedForward(embed_dim) 
        self.ln3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(D_PROB)
    
    def forward(self, x, enc: torch.Tensor, padding_mask, enc_padding_mask):
        # assert x.size() == enc.size(), f"Encoder output and decoder sublayer 1 output must be same shape: {enc.size()} {x.size()}"
        x = self.mh1(x, x, x, padding_mask)
        x = self.ln1(x + self.dropout(x))
        x = self.mh2(enc, x, enc, enc_padding_mask)
        x = self.ln2(x + self.dropout(x))
        x = self.ff(x)
        x = self.ln3(x + self.dropout(x))
        return x


# ---------------------------------------------------------------------------------------------
#                             (More Important) CLASSES
# ---------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, N: int, num_heads: int, embed_dim: int, vocab_size: int, context: int):
        super().__init__()
        self.in_emb = nn.Embedding(vocab_size, embed_dim) # (vocabsize x embed_dim) each token/word will have a 512 dim representation
        self.out_emb = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.ModuleList([EncoderBlock(num_heads, embed_dim) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderBlock(num_heads, embed_dim) for _ in range(N)])
        self.lin = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(D_PROB)
        self.register_buffer('pos_enc', self.pos_encoding(context, embed_dim))
        self.apply(self._init_weights)
    
    def forward(self, src, trg, src_mask, trg_mask):
        x = self.decode(trg, self.encode(src, src_mask), trg_mask, src_mask)
        x = self.lin(x)
        return x
    
    # better to separate encode and decode so we can see what's happening downstream during inference
    def encode(self, x, padding_mask):
        x = self.in_emb(x)
        _, T, _ = x.size()
        x = self.dropout(x + self.pos_enc[:T])
        for encoder in self.encoder:
          x = encoder(x, padding_mask)
        return x
    
    def decode(self, x, src, padding_mask, enc_padding_mask):
        x = self.out_emb(x)
        _, T, _ = x.size()
        x = self.dropout(x + self.pos_enc[:T])
        for decoder in self.decoder:
            x = decoder(x, src, padding_mask, enc_padding_mask)
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
    def generate(self, sentence, start_token_idx: int, max_tokens, greedy=True):
        """Generation of text from an input given that we are greedy decoding or using a multinomial"""
        assert self.training, "Transformer model should be in training mode"
        
        # check shapes of sentence, put start token into torch array
        if len(sentence.shape) < 2 and isinstance(sentence, torch.Tensor):
            sentence = sentence[None, :]
        else:
            assert isinstance(sentence, torch.Tensor), "Input sentence is not a torch tensor"
            assert len(sentence.shape) <= 2, f"Input sentence has more than 2 dimensions, has {len(sentence.shape)}"
            
        src = sentence
        trg = torch.tensor([[start_token_idx]])
        
        for _ in range(max_tokens):
            logits = self(src, trg)
            logits = logits[:, -1, :] # final time step
            probs = F.softmax(logits, dim=-1)
            if greedy:
                probs = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                probs = torch.multinomial(probs, num_samples=1)
            trg = torch.cat((trg, probs), dim=1)
    
    def _init_weights(self, module):
        pass
    
class Paraphrase(Dataset):
    def __init__(self, src_dir, trg_dir, nlp):
        """Initialization, we need a directory to load as well as tokenization framework"""
        start = time.time()
        self.nlp = nlp
        src = self.open(src_dir)
        trg = self.open(trg_dir)
        data_pairs = list(zip(src, trg))
        
        # by itself this will take ~90 seconds, there's no real way to get around this besides compute
        self.data = [(self.tokenize(doc), self.tokenize(summary)) for doc, summary in data_pairs]
        
        # just caching to make it much easier to access, save computation if needed
        self._map()
        self._max_context = max([len(src) for src, _ in self.data])
        end = time.time()
        print(f"Total time for dataset initialization was {end - start}")

    def __len__(self):
        """Length method that is needed for PyTorch DataLoader"""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item method that is also needed to PyTorch's DataLoader"""
        src, trg = self.data[idx]
        src = torch.tensor([self.stoi[s] for s in src])
        trg = torch.tensor([self.stoi[t] for t in trg])
        return (src, trg)

    def _map(self):
        """Creates a string-to-index and index-to-string based on sorted vocab"""
        self._vocabulary()
        self.stoi = {v:i for v, i in zip(self.vocab, range(self.vocab_size()))}
        self.itos = {i:v for v, i in zip(self.vocab, range(self.vocab_size()))}
    
    def map(self, map="both"):
        """Getter method if we want to double check stoi and itos"""
        try:
            if map == "itos":
                return self.itos
            elif map == "stoi":
                return self.stoi
            elif map == "both":
                return (self.itos, self.stoi)
            else:
                raise
        except Exception as e:
            raise AttributeError(f"Takes stoi, itos, or both. You gave {map}")

    def get_pad_id(self):
        return self.stoi["<p>"]

    # need to find the max length so that we can pad accordingly
    def _vocabulary(self):
        """Finding the unique words in the train and test set together for itos and stoi"""
        train_set = set([token for train, _ in self.data for token in train])
        test_set = set([token for _, test in self.data for token in test])
        t = train_set | test_set
        t.add("<p>")
        self.vocab = sorted(t)

    def vocab(self):
        """Getter method to check the vocab if we need it"""
        return self.vocab

    def vocab_size(self):
        """Getter method for the vocab size"""
        return len(self.vocab)

    def max_context(self):
        """Getter method for returning the max context"""
        return self._max_context

    def tokenize(self, text):
        """Tokenization utilizing spaCy's tokenzation in list comprehension; tokens manually added since it's easier"""
        t = [token.text.lower() for token in self.nlp.tokenizer(text)]
        # it's easier if we manually add the tokens after spaCy's tokenization so we have more granularity
        return ["<s>"] + t + ["<e>"]

    def open(self, path):
        """Opening the files using utf-8"""
        with open(path, 'r', encoding='utf-8') as file:
            file = file.readlines()
            file = [line.strip()for line in file]
        return file
    
class CustomOptimizer:
    """
    Not my original idea, got it from a SO which linked to a GitHub
    TODO: Find the SO to the GitHub as to not plagarize
    """
    def __init__(self, optmizer, d_model, warmup_steps):
        self._optimizer = optmizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0
        
    def zero_grad(self):
        self._optimizer.zero_grad()
        
    def step(self):
        self._update_lr
        self._optimizer.step()
        
    def state_dict(self):
        return self._optimizer.state_dict()
    
    def get_step(self):
        return self._step
        
    def _update_lr(self):
        self._step += 1
        
        right = min(self._step**-0.5, self._step * self.warmup_steps**-1.5)
        lr = (self.d_model**-0.5) * right
        for g in self._optimizer.param_groups:
            g['lr'] = lr 


# ---------------------------------------------------------------------------------------------
#                                   FUNCTIONS
# ---------------------------------------------------------------------------------------------

def custom_collate_fn(batch):
    """Goes into the DataLoader so that every sentence is padded correctly"""
    train = [t for t, _ in batch]
    test = [t for _, t in batch]
    
    pad_src = pad_sequence(train, batch_first=True, padding_value=2683)
    pad_trg = pad_sequence(test, batch_first=True, padding_value=2683)

    return pad_src, pad_trg

def checkpoint(model, optimizer, batch, epoch):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "batch": batch
    }, f"models/optimus_checkpoints_epoch{epoch}_batch{batch}.pth")
    
def make_padding_mask(input_ids, padding_token_id):
    mask = (input_ids == padding_token_id).type(torch.float32)
    return mask