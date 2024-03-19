# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Transformer, Paraphrase, custom_collate_fn
from tqdm import tqdm
import spacy
import time

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
BETA_1 = 0.9
BETA_2 = 0.98
EPISILON = 1e-9
EPOCHS = 10
WARMUP_STEPS = 4000
LAYERS = 6
HEADS = 8
EMBED_DIM = 512

spacy_en = spacy.load('en_core_web_sm')
src_file_path = "../../data/paraphrases/train/train.src"
tgt_file_path = "../../data/paraphrases/train/train.tgt"

paraphrase_data = Paraphrase(src_file_path, tgt_file_path, spacy_en)
VOCAB_SIZE = paraphrase_data.vocab_size()
CONTEXT = paraphrase_data.max_context()
loader = DataLoader(paraphrase_data, batch_size=64, collate_fn=custom_collate_fn, shuffle=True)

optimus = Transformer(LAYERS, HEADS, EMBED_DIM, VOCAB_SIZE, CONTEXT).to(device)
optimus.train()

optimizer = optim.Adam(optimus.parameters(), lr=LR, betas=(BETA_1, BETA_2), eps=EPISILON)

# we will have to F.crossentropy to get the loss, we can still do loss.backwards() check the docs
for _ in EPOCHS:
    for src, trg in loader:
        B, T = trg.size() 
        total_loss = 0
        for i in range(1, T):
            y_hat = optimus(src, trg[:, :i]) # source isn't changing, we are teacher forcing otherwise it would take much longer to train
            loss = F.binary_cross_entropy(y_hat, trg[:, i+1]) # we show the next word, remember our output is what the next token will be not the entire sentence!