from src.autoencoder import Autoencoder

import numpy as np
import sklearn
import torch

class MixedManifoldDetector:
    """
    
    """
    def __init__(self, autoencoder: Autoencoder = None, manifold_alg: sklearn.base.TransformerMixin = None):
        if autoencoder is None:
            self.autoencoder = Autoencoder()
        else:
            self.autoencoder = autoencoder

        if manifold_alg is None:
            self.manifold_alg = TSNE()
        else:
            self.manifold_alg = manifold_alg

    """
    
    """
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        ...

    """
    
    """
    def fit(self, train_data: np.ndarray):
        ...
    
    """
    
    """
    def transform(self, data: np.ndarray) -> np.ndarray:
        ...
