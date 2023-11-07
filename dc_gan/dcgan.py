# import dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dcmodels import Discriminator, Generator, CatFaces
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNEL_IMG = 3 # check this
Z_DIM = 100
EPOCHS = 50
BETA_1 = 0.5
FEATURES_DISC = 64
FEATURES_GEN = 64

# datapreprocessing
transformation = transforms.Compose(
[
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNEL_IMG)], [0.5 for _ in range(CHANNEL_IMG)]
    )
]
)
dataset = CatFaces(targ_dir="../../data/CatFaces/", transform=transformation)
print(f"Length of the dataset: {len(dataset)}")
loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

# setting up or models, initialization is in the subclass in dcmodels.py
gen = Generator(Z_DIM, FEATURES_GEN, CHANNEL_IMG).to(device)
disc = Discriminator(FEATURES_DISC, CHANNEL_IMG).to(device)
gen.train()
disc.train()

# losses and optimization methods
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.999))
loss = nn.BCELoss()
    
# fixed noise to track progress through our training
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

# Tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0


# training loop
for epoch in range(EPOCHS):
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for batch_idx, real in loop:        
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        
        ## Train Disc to maximize log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1) # N x 1 x 1 x 1 -> N
        loss_disc_real = loss(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        opt_disc.zero_grad()
        loss_disc.backward(retain_graph=True) # retain graph true since we are reutilizing this fake in second part but pytorch will remove intermediates
        opt_disc.step()
        
        ### Train the Gen to maximize log(D(G(z))) aka tricking the discriminator
        output = disc(fake).reshape(-1)
        loss_gen = loss(output, torch.ones_like(output))
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        if batch_idx % 100 == 0:
            with torch.no_grad(): # no computational graphs to keep track of how we're doing
                fake = gen(fixed_noise)
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
            step += 1
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

torch.save({
    'gen_state_dict':gen.state_dict(),
    'disc_state_dict':disc.state_dict()
}, 'DCGAN.pth')

"""
Load:
    modelA = TheModelAClass(*args, **kwargs)
    modelB = TheModelBClass(*args, **kwargs)
    optimizerA = TheOptimizerAClass(*args, **kwargs)
    optimizerB = TheOptimizerBClass(*args, **kwargs)

    checkpoint = torch.load(PATH)
    modelA.load_state_dict(checkpoint['modelA_state_dict'])
    modelB.load_state_dict(checkpoint['modelB_state_dict'])
    optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
    optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])

    modelA.eval()
    modelB.eval()
    # - or -
    modelA.train()
    modelB.train()
"""