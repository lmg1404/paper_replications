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