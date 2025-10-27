"""
Autoencoder variacional
"""
import numpy as np
import torch
import torch.nn as nn
from autoencoder import Autoencoder


class VariationalAutoencoder(Autoencoder):
    """
    Representa un autoencoder variacional.
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
        super(VariationalAutoencoder, self).__init__(batch_size, input_dim, latent_dim,
                                                     lr, epochs, loss_fn, error_threshold, device)

    def _build_encoder(self) -> None:
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(in_features=64, out_features=self.latent_dim)
        self.logvar_layer = nn.Linear(
            in_features=64, out_features=self.latent_dim)

    def _build_decoder(self) -> None:
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.input_dim)
        )

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Aplica el "reparametrization trick" para muestrear de la distribución latente.
        Args:
            mu (torch.Tensor): Media de la distribución.
            logvar (torch.Tensor): Logaritmo de la varianza de la distribución.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Aplicamos reparametrization trick

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implementación de Autoencoder._forward que incluye sampling utilizando
        el "reparametrization trick".
        Args:
            x (torch.Tensor): Datos de entrada.
        Output:
            Representación latente de los datos y reconstrucción de los datos.
        """
        encoder_output = self.encoder(x)
        mu = self.mu_layer(encoder_output)
        logvar = self.logvar_layer(encoder_output)
        z = self._sample(mu, logvar)
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
        if not self.trained:
            return None

        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(
                data, dtype=torch.float32, device=self.device)
            encoder_output = self.encoder(data_tensor)
            mu, logvar = self.mu_layer(
                encoder_output), self.logvar_layer(encoder_output)

            z = self._sample(mu, logvar)
        return z.cpu().numpy()
