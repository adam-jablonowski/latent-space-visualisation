
import pickle

import numpy as np
from sklearn.decomposition import PCA
import torch
from sklearn.neighbors import KDTree


class Neighbours:

    def __init__(self, points=None, repr=None):
        if points is not None:
            self.tree = KDTree(points, leaf_size=2)
        elif repr is not None:
            self.tree = pickle.loads(bytes(repr))

    def neighbours_idxs(self, points, radius):
        return self.tree.query_radius(points, r=radius)

    def to_string(self):
        return list(pickle.dumps(self.tree))
    