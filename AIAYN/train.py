# import dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Transformer
from tqdm import tqdm

# hyper parameters
BETA_1 = 0.9
BETA_2 = 0.98
EPISILON = 1e-9
EPOCHS = 20
WARMUP_STEPS = 4000
