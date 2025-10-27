"""
Autoencoder lineal
"""

import torch
from autoencoder import Autoencoder


class LinearAutoencoder(Autoencoder):
    """
    Representa un autoencoder lineal.
    """

    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 loss_fn: torch.nn.Module | None = None,
                 error_threshold: float = 0.0,
                 device: str = "cpu"
                 ):
        """
        Constructor de la clase LinearAutoencoder.
        Args:
            batch_size (int): Tamaño de los batches para el entrenamiento.
            input_dim (int): Dimensión de los datos de entrada.
            latent_dim (int): Dimensión de la capa de embedding.
            lr (float): Learning rate a utilizar en la optimización de parámetros.
            epochs (int): Número de epochs para el entrenamiento.
            loss_fn (torch.nn.Module): Función de pérdida a utilizar para el entrenamiento.
            error_threshold (float): Umbral para detener el entrenamiento dependiendo del
                                     error de reconstrucción.
            device (str): Dispositivo en que entrenar el modelo.
        """
        super(LinearAutoencoder, self).__init__(batch_size, input_dim, latent_dim,
                                                lr, epochs, loss_fn, error_threshold, device)

    def _build_encoder(self):
        """
        Construye el encoder
        """
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_dim, out_features=128),
            torch.nn.Linear(in_features=128, out_features=64),
            torch.nn.Linear(in_features=64, out_features=self.latent_dim)
        )

    def _build_decoder(self):
        """
        Construye el decoder
        """
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.latent_dim, out_features=64),
            torch.nn.Linear(in_features=64, out_features=128),
            torch.nn.Linear(in_features=128, out_features=self.input_dim)
        )
