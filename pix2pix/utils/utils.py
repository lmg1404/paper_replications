import torch
from torchvision.utils import save_image
import numpy.random as random

def train_step(gen, disc, opt_disc, opt_gen, device, loop, bce, l1, LAMBDA, epoch, EPOCHS):
    for _, (y, x) in enumerate(loop):
        # real and base set to device or it will crash, no need for latent space here
        x = x.to(device)
        y = y.to(device)
        
        # run the base through generator: fake = gen(base)
        z = gen(x)
        
        # pass real through discriminator (30x30 I think is output)
        disc_real = disc(y, x)
        # pass fake through discriminator
        disc_fake = disc(z.detach(), x) # was this the problem child?
        
        # show the discriminator all 1s to maximize
        loss_disc_real = bce(disc_real, torch.ones_like(disc_real))
        # show discriminator all 0s to minimize
        loss_disc_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        
        # add then divide by 2 to slow training
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        
        # train the generator to MAXIMIZE log(D(G(base)))
        # pass fake through disc one more time
        gen_fake = disc(z, x)
        # how the generator 1's to maximize
        gen_bce = bce(gen_fake, torch.ones_like(gen_fake))
        gen_reg = l1(z, y) * LAMBDA 
        loss_gen = gen_bce + gen_reg
        
        # zero grad -> backward -> step etc etc
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
                
        loop.set_postfix({"L1": gen_reg.item(), "BCE": gen_bce.item(), "Disc": loss_disc.item()})
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")

def save_checkpoint(gen, disc, disc_optimizer, gen_optimizer, epoch):    
    torch.save({
        'gen_state_dict':gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'disc_optimizer': disc_optimizer.state_dict(),
        'gen_optimizer': gen_optimizer.state_dict(),
        'epoch': epoch
    }, f'Pix2Pix_{epoch}.pth')
    
def save_images(gen, validation, device, epoch, l1, LAMBDA):
    j = random.randint(0, len(validation))
    s = 0
    with torch.no_grad():
        for i, (y, x) in enumerate(validation):
            x, y = x[None, :], y[None, :]
            x = x.to(device)
            y = y.to(device)
            z = gen(x)
            s += l1(z, y) * LAMBDA 
            if i == j:
                x, y, z = x[0], y[0], z[0]
                image = torch.cat((x, y, z), dim=-1)
                save_image(image, f"images/epoch{epoch+1}.png")
        print(f"Average validation loss using L1 on the images was: {s/len(validation)}")
        
    
    