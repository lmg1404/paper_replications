# import dependencies
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, Generator
from tqdm import tqdm

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
LEARNING_RATE = 2e-4            # same as dc gan
BATCH_SIZE = 1                  # 1-10 depending on what we're doing
IMAGE_SIZE = 256
CHANNEL_IMG = 3                 # check this
EPOCHS = 50                     # anywhere from 50-75 would be good I think, early stopping is an option through board
BETA_1 = 0.5
BETA_2 = 0.999
LAMBDA = 100
FEATURES_DISC = 64
FEATURES_GEN = 64


# TODO: import dataset + preprocessing
# TODO: select 4 images before turning into a generator object
# TODO: turn datset object into data loader

# TODO: initialize the generator and discriminator also load to device
# TODO: set both gen and disc to train

# TODO: use Adam optimizer for discriminator and generator

# TODO: find out how to use the custom loss function using  L1 loss

# TODO: select 4 images to show how model "learns" through epochs

# TODO: use Tensorboard SummaryWriter, maybe make it neater this time

# TODO: training for loop here
    # TODO: tqdm, this was a good idea from last time
    
    # TODO: go into our small batches for training
    # TODO: something like for batch_idx, real, base in loop
    
        # TODO: real and base set to device or it will crash, no need for latent space here
        
        # TODO: run the base through generator: fake = gen(base)
        
        # TODO: pass real through discriminator (30x30 I think is output)
                # we might have to resize for computation
        # TODO: show the dicriminator all 1s to maximize
        # TODO: pass fake through discriminator
        # TODO: show discriminator all 0s to minimize
        # TODO: add the  L1 loss with gamma????????
        # TODO: add then divide by 2 to slow training
        # TODO: zero grad -> backward -> step etc etc
        
        # train the generator to MAXIMIZE log(D(G(base)))
        # TODO: pass fake through disc one more time
        # TODO: show the generator 1's to maximize
        # TODO: zero grad -> backward -> step etc etc
        
        # TODO set board if statement for batch idx
            # TODO: wrapper no grad
                # TODO: fake the 4 images from above
                # TODO: board the make grid for real
                # TODO: board the make grid for fake
            
            # TODO: add a step
        # TODO: set description for our epochs