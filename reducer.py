import pickle

import numpy as np
from sklearn.decomposition import PCA


class Reducer:
    def __init__(self, points=None, repr=None):
        if points is not None:
            self.pca = PCA(n_components=2)
            self.pca.fit(points)
            self.dim = len(points[0])
        elif repr is not None:
            self.pca = pickle.loads(bytes(repr[0]))
            self.dim = repr[1]

    def reduce(self, points):
        return self.pca.transform(points)

    def reconstruct(self, points):
        return self.pca.inverse_transform(points)

    def reconstruction_quality(self, points):
        reconstructed = self.reconstruct(self.reduce(points))
        return np.array([np.linalg.norm(p - r) for p, r in zip(points, reconstructed)])

    def to_string(self):
        return (list(pickle.dumps(self.pca)), self.dim)
