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

Summarize the results of your replication. Include metrics, comparisons with the original paper (if available), and any insights gained during the process.

### Usage

Provide instructions on how to run or reproduce your implementation. Include any dependencies, setup instructions, and example commands.