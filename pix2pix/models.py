import torch.nn as nn 
import torch
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, features:int = 64, img_channels:int = 3):
        super(Generator, self).__init__()
        
        # down
        self.down1 = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.down2 = self._down_block(features, features*2, 4, 2, 1)
        self.down3 = self._down_block(features*2, features*4, 4, 2, 1)
        self.down4 = self._down_block(features*4, features*8, 4, 2, 1)
        self.down5 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down6 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down7 = self._down_block(features*8, features*8, 4, 2, 1)
        self.down8 = self._down_block(features*8, features*8, 4, 2, 1)
        
        # up
        self.up1 = self._up_block(features*8, features*8, 4, 2, 1)
        self.up2 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up3 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up4 = self._up_block(features*16, features*8, 4, 2, 1)
        self.up5 = self._up_block(features*16, features*4, 4, 2, 1)
        self.up6 = self._up_block(features*8, features*2, 4, 2, 1)
        self.up7 = self._up_block(features*4, features, 4, 2, 1)
        self.final = nn.Sequential(
            nn.Conv2d(features*2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    
    def _down_block(self, in_channels, out_channels, kernel, stride, padding): 
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, bias=False),
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
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        
        # up
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        final = self.final(torch.cat([u7, d1], dim=1))
        
        return final
        



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
        
    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

    def _block(self, in_channels, out_channels, kernel, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
def test():
    torch.cuda.empty_cache()
    device = "cuda"
    x = torch.randn((1, 3, 256, 256))
    gen = Generator().to(device)
    disc = Discriminator()
    print(summary(gen, (3, 256, 256)))
    # print(f"Generator output shape: {gen(torch.rand((1, 3, 256, 256)).to(device)).size()}")
    print(f"Discriminator output shape: {disc(x, x).size()}")

if __name__ == "__main__":
    test()