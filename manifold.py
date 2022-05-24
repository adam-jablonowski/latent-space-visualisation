
import time

import numpy as np
import torch
import torchvision
from torch.autograd import functional
from torchvision import datasets, transforms
import torch.nn.functional as F

import torch.nn as nn
import plotly.graph_objects as go

class Manifold:


    def measure(self, z):
        ### Compute the measure of curvature
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.linalg.det(M))
    
    def loss_function(self, *args):
        ### Computes loss of model
        raise NotImplementedError()

    def encode(self, x):
        ### Returns vectorized encoding of x as mu and log var
        raise NotImplementedError()

    def decode(self, z):
        ### Returns vectorized encoding of sampled z
        raise NotImplementedError()
    
    def get_datasets(self, batch_size = 100):
        ### Returns tuple of datasets train and test loader
        raise NotImplementedError()

    def point_info(self, point):
        ### Returns info about latent point in form of go.Image
        raise NotImplementedError()
    
    def metric_tensor(self, z, nargout=1):
        ### Returns vectorized metric tensor at latent point z (and its jacobian if nargout=2)
        raise NotImplementedError()    
