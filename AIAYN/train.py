# import dependencies
print("Importing dependencies")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Transformer, Paraphrase, custom_collate_fn, CustomOptimizer, checkpoint
from tqdm import tqdm
import spacy
import time

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters
LR = 0.0001
BETA_1 = 0.9
BETA_2 = 0.98
EPISILON = 1e-9
EPOCHS = 10
WARMUP_STEPS = 4000
LAYERS = 4
HEADS = 4
EMBED_DIM = 256

# loading a preparing the data
print("Loading the data")
spacy_en = spacy.load('en_core_web_sm')
src_file_path = "../../data/paraphrases/train/train.src"
tgt_file_path = "../../data/paraphrases/train/train.tgt"
paraphrase_data = Paraphrase(src_file_path, tgt_file_path, spacy_en)
VOCAB_SIZE = paraphrase_data.vocab_size()
CONTEXT = paraphrase_data.max_context()
loader = DataLoader(paraphrase_data, batch_size=16, collate_fn=custom_collate_fn, shuffle=True)

# getting our transformer named optimus
print("Setting up model and getting the custom optimizer")
optimus = Transformer(LAYERS, HEADS, EMBED_DIM, VOCAB_SIZE, CONTEXT).to(device)
optimus.train()

# set up or optimizer such that we have our learning rate set
adam = optim.Adam(optimus.parameters(), lr=LR, betas=(BETA_1, BETA_2), eps=EPISILON)
optimizer = CustomOptimizer(adam, EMBED_DIM, WARMUP_STEPS)

# TensorBoard
writer = SummaryWriter(f"logs/loss")

# we will have to F.crossentropy to get the loss, we can still do loss.backwards() check the docs
step = 0
print("Starting training")
for epoch in range(EPOCHS):
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch_idx, (src, trg) in loop:
        src = src.to(device)
        trg = trg.to(device)
        B, T = trg.size() 
        total_loss = 0
        for i in range(1, T):
            pred = optimus(src, trg[:, :i]) # source isn't changing, we are teacher forcing otherwise it would take much longer to train
            pred = pred[:, -1, :] # focus on the last time step?
            loss = F.cross_entropy(pred, trg[:, i]) # we show the next word, remember our output is what the next token will be not the entire sentence!
            total_loss += loss
        avg_loss = total_loss / T
        
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        if batch_idx % 5 == 0:
            writer.add_scalar("Loss", avg_loss, global_step=step)
            step += 1
        
        if batch_idx % 50 == 0:
            checkpoint(optimus, optimizer, batch_idx, epoch)
        
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
    