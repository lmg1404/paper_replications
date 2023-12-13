# import dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator, Facades
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 2e-4            # same as dc gan
BATCH_SIZE = 4                  # 1-10 depending on what we're doing
IMAGE_SIZE = 256
CHANNEL_IMG = 3                 # check this
EPOCHS = 75                     # anywhere from 50-75 would be good I think, early stopping is an option through board
BETA_1 = 0.5
BETA_2 = 0.999
LAMBDA = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

    
# import dataset + preprocess already inside the object, maybe for inference we make another object?
# depending on what directory we run from we use one or the other
# dataset = Facades(targ_dir="../data/facades/")
dataset = Facades(targ_dir="data/facades/")
print(f"Length of dataset: {len(dataset)}")
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)


# initialize our models and set them to train
gen = Generator(features=FEATURES_GEN, img_channels=CHANNEL_IMG).to(device)
disc = Discriminator(features=FEATURES_DISC, img_channels=CHANNEL_IMG).to(device)
gen.train()
disc.train()


# use Adam optimizer for discriminator and generator
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))


# initialize loss objects, just have to add them up later and do backwards
bce = nn.BCELoss()
l1 = nn.L1Loss()


# TODO: use Tensorboard SummaryWriter, maybe make it neater this time
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")


for epoch in range(EPOCHS):
    # tqdm, this was a good idea from last time
    loop = tqdm(loader, total=len(loader), leave=False)
    
    # go into our small batches for training
    for batch_idx, (y, x) in enumerate(loop):
        # real and base set to device or it will crash, no need for latent space here
        y = y.to(device)
        x = x.to(device)
        
        
        # run the base through generator: fake = gen(base)
        z = gen(x)
        #pass real through discriminator (30x30 I think is output)
        disc_real = disc(x, y)
        # show the dicriminator all 1s to maximize
        loss_disc_real = bce(disc_real, torch.ones_like(disc_real))
        # pass fake through discriminator
        disc_fake = disc(x, z)
        # show discriminator all 0s to minimize
        loss_disc_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        # add then divide by 2 to slow training
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        
        # zero grad -> backward -> step etc etc
        opt_disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()
        
         
        # train the generator to MAXIMIZE log(D(G(base)))
        # pass fake through disc one more time
        gen_fake = disc(x, z)
        # how the generator 1's to maximize
        gen_bce = bce(gen_fake, torch.ones_like(gen_fake))
        # add the  L1 loss with gamma????????
        gen_reg = l1(z, x) * LAMBDA 
        
        # zero grad -> backward -> step etc etc
        loss_gen = gen_bce + gen_reg
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        # TODO set board if statement for batch idx
            # TODO: wrapper no grad
                # TODO: fake the 4 images from above
                # TODO: board the make grid for real
                # TODO: board the make grid for fake
            
            # TODO: add a step
        # TODO: set description for our epochs