import torch as th
import torch.nn as nn
import dgl
import math

class DGI(nn.Module):
    # encoder = MAGNN_gc_dgi
    def __init__(self, encoder, n_hidden):
        super().__init__()
        self.encoder = encoder
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    # @param blocks: list( a graph as a block )
    def forward(self, blocks, features):
        perm = th.randperm(features.shape[0])
        corrupted_features = features[perm]

        positive = self.encoder(blocks, features)
        negative = self.encoder(blocks, corrupted_features)
        s = th.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, s)
        negative = self.discriminator(negative, s)

        l1 = self.loss(positive, th.ones_like(positive))
        l2 = self.loss(negative, th.zeros_like(negative))

        return l1 + l2

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.weight = nn.Parameter(th.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = th.matmul(features, th.matmul(self.weight, summary))
        return features
