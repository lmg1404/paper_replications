import torch.nn as nn 
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self, features:int = 64, img_channels:int = 3):
        super(Generator, self).__init__()
        
        # down
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down2 = self._down_block(features, features*2, 4, 2, 1)
        self.down3 = self._down_block(features*2, features*4, 4, 2, 1)
        self.down4 = self._down_block(features*4, features*8, 4, 2, 1)
        self.down5 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down6 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down7 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down8 = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        # up
        self.up1 = self._up_block(features*8, features*8, 4, 2, 1)
        self.up2 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up3 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up4 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up5 = self._up_block(features*16, features*4, 4, 2, 1)
        self.up6 = self._up_block(features*8, features*2, 4, 2, 1)
        self.up7 = self._up_block(features*4, features, 4, 2, 1)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # apply initialization
        # NOTE: self.apply(fn) does so inplace
        # NOTE: self._apply(fn) is NOT inplace and returns an object
        self.apply(self._init_weights)
    
    
    def _down_block(self, in_channels, out_channels, kernel, stride, padding): 
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def _up_block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.5),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        # down
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # up
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        final = self.final(torch.cat([u7, d1], dim=1))
        
        # print(f"u1: {u1.size()}")
        # print(f"u2: {u2.size()}")
        # print(f"u3: {u3.size()}")
        # print(f"u4: {u4.size()}")
        # print(f"u5: {u5.size()}")
        # print(f"u6: {u6.size()}")
        # print(f"u7: {u7.size()}")
        # print(f"final: {final.size()}")
        return final
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(tensor=module.weight, mean=0, std=0.02)



class Discriminator(nn.Module):
    def __init__(self, features:int = 64, img_channels:int = 3):
        super(Discriminator, self).__init__() 
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels*2, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.layers = nn.Sequential(
            self._block(features, features*2, 4, 2, 1),
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 1, 1)
        )
        self.final = nn.Sequential(
            nn.Conv2d(features*8, 1, 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.model = nn.Sequential(self.initial, self.layers, self.final)
        
        # apply initialization in place
        self.apply(self._init_weights)
        
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(tensor=module.weight, mean=0, std=0.02)
        
        
class Facades(Dataset):
    
    def __init__(self, targ_dir: str) -> None:
        self.paths = list(Path(targ_dir).glob("*/*")) 
        self.transforms1 = transforms.Compose(
            [transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)]
            )]
        )
        self.transforms2 = transforms.Compose([
            transforms.Resize(size=(286, 286)),
            transforms.RandomCrop(size=(256, 256)),
        ])
    
    def load_image(self, idx: int) -> Image.Image:
        image_path = self.paths[idx]
        return Image.open(image_path)
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.load_image(idx)
        
        img_ten = self.transforms1(img)

        real = img_ten[:, :, :256]
        input = img_ten[:, :, 256:]

        real = self.transforms2(real)
        input = self.transforms2(input)

        return real, input

def test():
    torch.cuda.empty_cache()
    x = torch.randn((1, 3, 256, 256))
    gen = Generator()
    disc = Discriminator()
    print(f"Generator output shape: {gen(x).size()}")
    print(f"Discriminator output shape: {disc(x, x).size()}")



if __name__ == "__main__":
    test()