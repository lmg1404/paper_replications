## Image-to-Image Translation with Conditional Adversarial Networks (aka Pix2Pix)

### Overview

A conditional GAN: instead of taking a latent space input (i.e. random numbers) it takes some structured input, in this case another image. In my eyes this kind of lays the groundwork for the 
models we have today, though it's important to make the distinction that ones that we are used to today (DALL-E, StableDiffusion) are diffusion models (I'm not too sure the difference quite yet). 

### Implementation Details

- Programming Language: Python
- Deep Learning Framework: PyTorch
- Datasets: [Facades Data Set](https://cmp.felk.cvut.cz/~tylecr1/facade/)
- Key Modifications: None, going to add the capability to make a CLI for custom data sets in the future

### Results

Before we get into the details some **blunders** I had from my initial construction in November:
- L1 Loss was computed on the inputs (shapes) and the outputs of the GAN
    - *This shouldn't be the case it should be inputs, it would learn to implement 1's instead of weights to create the realistic looking facades*
- I had a dropount of 0.5 thoughout the entire decoder for the Generator
    - *According to the paper the first 3 'blocks' of the decoder use dropout*
    - *A dropout of 0.5 is super high since half the neurons of the decoder are supposed to make the image from the bottleneck plus the residual down blocks*
- Not training long enough
    - *I naively thought epochs of 10 or 20 would train this model since my experience is with DCGAN in the repository*
    - *It's extremely important to note the difference between both data sets where being close to 16k images allows for less epochs*
    - *The initial experiment in the paper uses 200 epochs which is now reflected in the repository*

### Usage

Provide instructions on how to run or reproduce your implementation. Include any dependencies, setup instructions, and example commands.