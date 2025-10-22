"""
Autoencoder lineal
"""

import torch
from autoencoder import Autoencoder


class LinearAutoencoder(Autoencoder):
    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 loss_fn: torch.nn.Module | None = None,
                 error_threshold: float = 0.0,
                 device: str = "cpu"):
        """

        """
        super().__init__(batch_size, input_dim, latent_dim,
                         lr, epochs, loss_fn, error_threshold, device)

    def _build_encoder(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_dim, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.latent_dim)
        )

    def _build_decoder(self):
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=self.input_dim)
        )
