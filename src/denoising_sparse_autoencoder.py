"""
Autoencoder con regularizaciones Denoising y Sparse
"""

import torch
from linear_sparse_autoencoder import LinearSparseAutoencoder

class DenoisingSparseAutoencoder(LinearSparseAutoencoder):
    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 loss_fn: torch.nn.Module | None = None,
                 error_threshold: float = 0.0,
                 device: str = "cpu",
                 lambda_val: float = 1e-3,
                 noise_factor: float = 0.3):
        """
        Constructor de la clase DenoisingSparseAutoencoder.
        Args:
            lambda_val (float): También llamado sparsity weight. Controla el peso de la regularización L1.
            noise_factor (float): Porcentaje de ruido que se introducirá a los datos de entrenamiento.
        """

        super().__init__(batch_size, input_dim, latent_dim, lr, epochs, loss_fn, error_threshold, device)
        self.lambda_val: float = lambda_val
        self.noise_factor: float = noise_factor

    def _add_noise(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Método que añade ruido al batch de datos. 
        Args:
            x_batch (torch.Tensor): Batch actual del entrenamiento.
        Output:
            Batch de datos con ruido añadido.
        """
        noisy_batch = x_batch + torch.randn_like(x_batch) * self.noise_factor
        if x_batch.min() >= 0.0 and x_batch.max() <= 1.0:  # Aplicar clipping si los datos están en [0.0, 1.0]
            noisy_batch = torch.clip(noisy_batch, 0.0, 1.0)
        return noisy_batch
