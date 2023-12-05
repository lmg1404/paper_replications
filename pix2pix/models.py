import torch.nn as nn 

class Generator(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(Generator, self).__init__()
        # TODO: figure out skip connections from down to up sampling in UNet
    
    
    def _down_block(self): # do the same thing we did in DC, it's very neat, but we have to figure out upsampling and downsampling
        return nn.Sequential(
            nn.Conv2d(),
            nn.BatchNorm2d(),
            nn.LeakyReLU(0.2)
        )
    
    def _up_block(self):
        pass
        



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__() 

    def _block(self):
        pass