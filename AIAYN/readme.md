## Attention is All You Need

### Overview

2017 Paper introducing the Transformer by Vaswani et. al.

### Implementation Details

- Programming Language: Python
- Deep Learning Framework: PyTorch
- Datasets: [Paraphrase Generation](https://github.com/dqxiu/ParaSCI/tree/master) or [Text Summarization](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
- Key Modifications: TBD

### Results

#### Attempt 1:
This model, named **Optimus**, was not very good. I padded the sequence lengths so PyTorch's `DataLoader` would not throw errors
whener the sequences different lengths, I did this using `torch.nn.utils.rnn.pad_sequence` which proved to be very effective. I just
need to know the padding sequence index. After training this model on 1500 batches, Google Colab would not let me train any longer
after 3 hours :(, but I'm putting checkpoints in after every 50 batches in case something like this happened.


Optimus only rememberd how to *pad*, unfortunately, since I did not account for the shorter sequences with more padding tokens.
I also seem to have not accounted for the test and train data sets may have different vocabularies, even though the 
`DataSet` object I constructed would go through the same sequences; though, in inference the mappings for both are the same, 
but when using `stoi` and `itos` on the test set I get complete gibberish which shouldn't be the case.

I'm not expectin the transformer to be the best thing in the world since it's from scratch and a pretty small architecture of only 4 encoder/decoder layers, 6 heads, and 192 embedded dimensions, but I don't think that this should remember how to only pad.

#### Attempt 2:
I will have to change how the loss function is calculated to avoid padding. I'll also increase the number of encoder/decoder layers
to 5 to see if that helps, initialize the weights (TBD), and keep everything else the same.

### Usage

Provide instructions on how to run or reproduce your implementation. Include any dependencies, setup instructions, and example commands.