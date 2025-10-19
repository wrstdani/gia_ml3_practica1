from autoencoder import Autoencoder

import torch
import torch.nn as nn

class SparseAutoencoder(Autoencoder):
    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 loss_fn: nn.Module | None = None,
                 error_threshold: float = 0.0,
                 optimizer: torch.optim.Optimizer | None = None,
                 device: str = "cpu"):
        super().__init__(batch_size, latent_dim, epochs, loss_fn, error_threshold, device)
    
    def _build_encoder(self):
        ...
    
    def _build_decoder(self):
        ...
