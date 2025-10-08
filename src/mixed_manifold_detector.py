"""
- Módulos propios:
"""
from src.autoencoder import Autoencoder

"""
- Librerías externas
"""
import numpy as np

import sklearn

class MixedManifoldDetector:
    """
    
    """
    def __init__(self, autoencoder: Autoencoder, manifold_alg: sklearn.base.TransformerMixin):
        ...
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
