import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class Generator(nn.Module):
    def __init__(self, zdim, features, img_dim):
        # based on what i found it will have to go like this:
        # Batch_size x z_dim(100) x 1 x 1, maybe this is the only way to make it work?
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            self._block(zdim, features*16, 4, 1, 0),
            self._block(features*16, features*8, 4, 2, 1),
            self._block(features*8, features*4, 4, 2, 1),
            self._block(features*4, features*2, 4, 2, 1),
            nn.ConvTranspose2d(features*2, img_dim, 4, 2, 1, bias=False), # why does the paper say 5??????
            nn.Tanh()
        )
        self.apply(self._init_weights)                                    # instantly applies weights to normal described to the certain modules
    
    def forward(self, x):
        return self.model(x)
    
    def _block(self, in_channels, out_channels, kernel, stride, padding): # again, using Aladdin's block since it's very nice to have and easy to read
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) # let gradient flows and fight dead neurons
        )
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(tensor=module.weight, mean=0, std=0.02)
            

class Discriminator(nn.Module):
    def __init__(self, features, img_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_dim, features*2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),                                      # remember, paper "...not applying batchnorm to generator output layer and discriminator input layer. [Sec 3]"
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 2, 1),
            self._block(features*8, features*16, 4, 2, 1),          # (-1, 1024, 4, 4) flatten dims 1024, 4, 4 into 16384
            nn.Conv2d(features*16, 1, 4, 2, 0, bias=False),                    
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
    
    def forward(self, x):
        return self.model(x)
    
    def _block(self, in_channels, out_channels, kernel, stride, padding): # gong to use Aladdin's block, it's much neater this way
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) # fighting dead neurons
        )
    
    def _init_weights(self, module):
        if isinstance(module, (nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(tensor=module.weight, mean=0, std=0.02)
                
class CatFaces(Dataset):
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        self.paths = list(Path(targ_dir).glob("*.jpg")) 
        self.transform = transform
    
    def load_image(self, idx: int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.load_image(idx)
        
        if self.transform:
            return self.transform(img)
        else:
            return img