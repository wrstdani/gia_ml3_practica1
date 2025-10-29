"""
Autoencoder lineal con regularización Sparse
"""

import torch
import torch.nn as nn
from linear_autoencoder import LinearAutoencoder


class LinearSparseAutoencoder(LinearAutoencoder):
    """
    Clase que representa un autoencoder lineal con regularización
    sparse (L1).
    """

    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 activation: nn.Module | None = None,
                 epochs: int = 100,
                 loss_fn: torch.nn.Module | None = None,
                 error_threshold: float = 0.0,
                 device: str = "cpu",
                 seed: int = 42,
                 lambda_val: float = 1e-3):
        """
        Constructor de la clase LinearSparseAutoencoder.
        Args:
            lambda_val (float): También llamado sparsity weight.
                                Controla el peso de la regularización L1.
        """
        super(LinearSparseAutoencoder, self).__init__(batch_size,
                                                      input_dim,
                                                      latent_dim,
                                                      lr,
                                                      activation,
                                                      epochs,
                                                      loss_fn,
                                                      error_threshold,
                                                      device,
                                                      seed)
        self.lambda_val: float = lambda_val

    def _compute_additional_loss(
        self,
        x_batch: torch.Tensor,
        z: torch.Tensor,
        recon: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula el término de regularización L1.
        Args:
            x_batch (torch.Tensor): Batch actual del entrenamiento.
            z (torch.Tensor): Embedding del batch actual del entrenamiento.
            recon (torch.Tensor): Output del modelo (por si quiere tenerse
                                   en cuenta el error de reconstrucción)
        Output:
            Término de regularización L1
        """
        return self.lambda_val * torch.mean(torch.abs(z))
