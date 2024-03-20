## Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Neural Networks

### Overview

A 2016 paper describing a new class of CNNs with the previous success of CNNs up to that point.
Purpose is to learn resuable feature representation from large *unlabeled* datasets. Employs both discriminator and generator like in other GANs.
Previous work on resuable feature representation is what the call the "classic approach" which uses clustering on data (K-Means).
CNN black-box critism is acknowledged since they can be difficult to visualize their inner workings for interpretability.

At this time researchers were having trouble scaling up GANs stably, and the previous year LAPGAN was developed to upscale low resolution generated images.
Researchers of this paper employed the following:
 1. The all convolotional net replaces maxpooling with strided convolutions, which is only in the generator
 2. Fully eliminating fully connected layers which are usually found at the top of convolutions
 3. Batch Normalization to stabilize learning ($\mu = 0$ and $\sigma = 1$)

The paper employs DCGAN was trained on 3 datasets, I assume 3 seperate models with 3 distinct parameters, on LSUN, Imagenet-1k, and a Faces dataset.
The datasets are used in the paper, but for this repository's purpose I used the [catfaces](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) dataset.
This one was used since we have no lables and a large amount of data, close to 16k 64x64 images.

### Implementation Details

- Programming Language: Python
- Deep Learning Framework: PyTorch
- Datasets: [CatFaces](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)
- Key Modifications: None

### Results

All in all compared to the diffusion models used today, this was a very simple network that I trained on my own laptop's NVIDIA GeForce RTX 3050 Ti Laptop GPU. The results from the  model are better than I expected them to be
after I followed the paper's specifications on the model exactly.
![25 images from the model](generated_images/25catfacesingrid.png)

The paper employs the use of traversing a dimension and seeing how it changes the model's output. I assumed that they modified the dimension by adding/subtracting so that's what I did in the inference notebook. In initial testing I found
that larger numbers means we can use less steps to see how this dimension changes the outputs.
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at0with10steps.png)
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at10with10steps.png)
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at13with10steps.png)
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at14with10steps.png)
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at26with10steps.png)
![Changes in cat faces by changing a dimension](generated_images/faceswithdelta3at6with10steps.png)

### Usage

Usage can be found at the end of `train.py` with the .pth file. All that's needed from this folder is `dcmodels.py` and `DCGAN.pth`.
All that needs to be done is load the generator into a variable using the same parameters as before and load the checkpoint from the .pth into the model.
Then the model needs to be set to evaluation as to avoid making a computational graph since we are not training anymore! We also need to load the .pth dictionary
into the GPU since we don't really need the GPU any more, this is a pretty small model creating small images:
```
CHANNEL_IMG = 3 
Z_DIM = 100
FEATURES_GEN = 64

checkpoint = torch.load("DCGAN.pth", map_location=torch.device('cpu'))

gen = Generator(Z_DIM, FEATURES_GEN, CHANNEL_IMG)
gen.load_state_dict(checkpoint["gen_state_dict"])
gen.eval();
```

A random image function was created as to create a single random noise tensor: 
```
def show_random_noise(generator, random_noise):
  img = torch.squeeze(gen(noise))
  img = (img - img.min()) / (img.max() - img.min())
  img = img * 255
  img = img.int()
  plt.imshow(img.permute(1, 2, 0).detach().numpy())
  plt.axis("off")
```

The rest of these functions and among other things can be found in the [inference notebook](inference.ipynb)
