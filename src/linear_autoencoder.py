"""
Autoencoder lineal
"""

import numpy as np
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
                 device: str | None = None,
                 seed: int = 42
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
                                                lr, epochs, loss_fn, error_threshold, device, seed)

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

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward del autoencoder (red neuronal).
        Args:
            x (torch.Tensor): Datos de entrada.
        Output:
            Embedding y reconstrucción de la entrada.
        """
        z = self.encoder(x)
        return (z, self.decoder(z))

    def _compute_additional_loss(
        self,
        x_batch: torch.Tensor,
        z: torch.Tensor,
        recon: torch.Tensor
    ) -> torch.Tensor:
        """
        Se utiliza para calcular un término adicional para la función de pérdida.
        Args:
            x_batch (torch.Tensor): Batch de datos de entrada.
            z (torch.Tensor): Representación latente del batch.
            recon (torch.Tensor): Reconstrucción del batch.
        """

        return torch.tensor(0.0, device=self.device)

    def _add_noise(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Se utiliza para añadir ruido en autoencoders con regularización denoising.
        Requiere que se implemente.
        Args:
            x_batch (torch.Tensor): Batch del conjunto de entrenamiento al que añadir ruido.
        Output:
            Batch de datos con ruido añadido.
        """
        return x_batch

    def transform(self, data: np.ndarray) -> np.ndarray | None:
        """
        Requiere que el autoencoder haya sido entrenado previamente
        con `fit()`. Se utiliza para obtener la representación latente
        (salida del encoder) del autoencoder de un conjunto de datos.
        Args:
            data (np.ndarray): Datos de los cuales se desea obtener el embedding.
        Output:
            Representación latente de la entrada.
        """
        if not self.trained:
            return None

        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(
                data, dtype=torch.float32, device=self.device)
            z = self.encoder(data_tensor)
        return z.cpu().numpy()
