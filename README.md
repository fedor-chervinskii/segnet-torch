# segnet-torch
Torch implementation of SegNet (http://arxiv.org/abs/1511.00561) and deconvolutional network (http://arxiv.org/abs/1505.04366)

## Note
For the moment, only batchSize = 1 is available because of pixel-wise loss. If there is real need for larger batches consider [nnx.MultiSoftMax](https://github.com/clementfarabet/lua---nnx#nnx.MultiSoftMax) module
