"""
- Módulos propios:
"""

"""
- Librerías externas
"""
import numpy as np

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    
    """
    def __init__(self, batch_size: int, epochs: int = 100, error_threshold: float = 0.0):
        self.epochs = epochs
        self.error_threshold = error_threshold

    """
    
    """
    def fit(self, data: np.ndarray):
        ...

    """
    
    """
    def transform(self, data: np.ndarray) -> np.ndarray:
        ...
