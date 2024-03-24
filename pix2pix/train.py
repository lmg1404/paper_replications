# import dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator, Facades
import matplotlib.pyplot as plt
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 2e-4            # same as dc gan
BATCH_SIZE = 1                  # facades was done with batch size of 1
IMAGE_SIZE = 256
CHANNEL_IMG = 3                 # check this
EPOCHS = 100                    # they trained the facades for 200 epochs, that's feasible on my machine
BETA_1 = 0.5
BETA_2 = 0.999
LAMBDA = 100
FEATURES_DISC = 64
FEATURES_GEN = 64

    
# import dataset + preprocess already inside the object, maybe for inference we make another object?
# need to be in ~/pix2pix/
dataset = Facades(targ_dir="../data/facades/train/", train=True)
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# below will be the selected batch we will put on the tensorboard
batch_selector = torch.randint(low=0, high=len(loader), size=(1,1)).item()
print(f"Length of dataset: {len(dataset)}")
print(f"Length of batches: {len(loader)}")
print(f"Batch number we will put on TensorBoard: {batch_selector}")

# initialize our models and set them to train
gen = Generator(features=FEATURES_GEN, img_channels=CHANNEL_IMG).to(device)
disc = Discriminator(features=FEATURES_DISC, img_channels=CHANNEL_IMG).to(device)
gen.train()
disc.train()


# use Adam optimizer for discriminator and generator
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))


# initialize loss objects, just have to add them up later and do backwards
bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()


# TODO: use Tensorboard SummaryWriter, maybe make it neater this time
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
writer_losses = SummaryWriter(f"logs/losses")

step = 0
for epoch in range(EPOCHS):
    # tqdm, this was a good idea from last time
    loop = tqdm(loader, total=len(loader), leave=True)
    # go into our small batches for training
    for batch_idx, (x, y) in enumerate(loop):
        # real and base set to device or it will crash, no need for latent space here
        x = x.to(device)
        y = y.to(device)
        
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
        gen_reg = l1(z, y) * LAMBDA 
        loss_gen = gen_bce + gen_reg
        
        # zero grad -> backward -> step etc etc
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
                
        loop.set_postfix({"L1": gen_reg.item(), "BCE": gen_bce.item(), "Disc": loss_disc.item()})
                
        # also going to plot loss to see how the model does over time
        if batch_idx % 8 == 0:
            # writer_losses.add_scalars("Architecture", {
            #     'Generator':loss_gen,
            #     'Discriminator':loss_disc
            # }, global_step=step)
            writer_losses.add_scalar("Generator Loss", loss_gen, step)
            step += 1

        # batch we put on the board
        if batch_idx == batch_selector:
            # no computational graph here
            with torch.no_grad():
                # fake the 4 images from above
                board_fake = gen(x)
                # board the make grid for real
                img_grid_real = torchvision.utils.make_grid(
                    y, normalize=True
                )
                # board the make grid for fake
                img_grid_fake = torchvision.utils.make_grid(
                    board_fake, normalize=True
                )
                
                writer_real.add_image("Real Facades", img_grid_real, global_step=epoch)
                writer_fake.add_image("Fake Facade", img_grid_fake, global_step=epoch)
            
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
    temp = gen(x)
    image = temp[0]
    save_image(image, f"images/epoch{epoch+1}.png")
        

# save our models for inference later on
# thinking about a computation combining pix2pix and WGAN but what else could be improved?
# last GAN I replicate for a while, video or NLP is next
torch.save({
    'gen_state_dict':gen.state_dict(),
    'disc_state_dict': disc.state_dict()
}, 'Pix2Pix.pth')