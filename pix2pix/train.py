# import dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from utils.models import Discriminator, Generator, Facades
from utils.utils import *
from tqdm import tqdm
import argparse

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# arg parsing to make sure we know what's going on
parser = argparse.ArgumentParser(description="Getting some hyper parameters")
parser.add_argument('--epochs', type=int, help='number of training loops to train', required=True)
parser.add_argument('--model_path', type=str, help='file name for previous models that require more training')
args = parser.parse_args()

# hyperparameters
LEARNING_RATE = 2e-4            # same as dc gan
BATCH_SIZE = 1                  # facades was done with batch size of 1
IMAGE_SIZE = 256
CHANNEL_IMG = 3                 # check this
EPOCHS = args.epochs            # they trained the facades for 200 epochs, that's feasible on my machine
BETA_1 = 0.5
BETA_2 = 0.999
LAMBDA = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

    
# import dataset + preprocess already inside the object, maybe for inference we make another object?
# need to be in ~/pix2pix/
dataset = Facades(targ_dir="../data/facades/train/")
validation = Facades(targ_dir="../data/facades/val/")
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize our models and set them to train
gen = Generator(features=FEATURES_GEN, img_channels=CHANNEL_IMG).to(device)
disc = Discriminator(features=FEATURES_DISC, img_channels=CHANNEL_IMG).to(device)
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

if args.model_path:
    checkpoint = torch.load(args.model_path)
    gen = gen.load_state_dict(checkpoint['gen_state_dict'])
    disc = disc.load_state_dict(checkpoint['disc_state_dict'])
    opt_gen = opt_gen.load_state_dict(checkpoint['gen_optimizer'])
    opt_disc = opt_disc.load_state_dict(checkpoint['disc_optimizer'])

gen.train()
disc.train()

# initialize loss objects, just have to add them up later and do backwards
bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

for epoch in range(EPOCHS):
    # tqdm, this was a good idea from last time
    loop = tqdm(loader, total=len(loader), leave=True)
    # go into our small batches for training
    train_step(gen, disc, opt_disc, opt_gen, device, loop, bce, l1, LAMBDA, epoch, EPOCHS)
    save_images(gen, validation, device, epoch, l1, LAMBDA)

    # thinking about a computation combining pix2pix and WGAN but what else could be improved?
    if epoch % 80 == 0:
        save_checkpoint(gen, disc, opt_disc, opt_gen, epoch)
save_checkpoint(gen, disc, opt_disc, opt_gen, epoch)