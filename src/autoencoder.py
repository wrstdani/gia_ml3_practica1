"""
Interfaz para construir diferentes tipos de autoencoders
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
from tqdm import tqdm


class Autoencoder(torch.nn.Module, ABC):
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
        Constructor base de la interfaz Autoencoder.
        Args:
            batch_size (int): Tamaño de los batches para
                              entrenar el autoencoder.
            input_dim (int): Tamaño de los datos de entrada
                             (número de características/dimensiones).
            latent_dim (int): Tamaño de la capa de embedding
                             del autoencoder (la salida del encoder).
            lr (float): Learning rate para descenso del gradiente
                        en el entrenamiento del autoencoder.
            epochs (int): Número de epochs de entrenamiento del autoencoder.
            loss_fn (torch.nn.Module): Función de pérdida para el entrenamiento del autoencoder.
            error_threshold (float): Umbral que determina si se continúa
                                     entrenando el autoencoder en cada epoch.
            device (str): Dispositivo en que PyTorch entrenará el autoencoder.
            seed (int): Semilla para poder replicar experimentos aleatorios
        """
        super(Autoencoder, self).__init__()

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.error_threshold = error_threshold
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.trained = False  # Para prevenir utilizar transform() sin entrenar
        self.encoder = None
        self.decoder = None
        self._build_architecture()
        self.to(self.device)

    @abstractmethod
    def _build_encoder(self) -> None:
        """
        Construye el encoder.
        """
        pass

    @abstractmethod
    def _build_decoder(self) -> None:
        """
        Construye el decoder.
        """
        pass

    def _build_architecture(self):
        """
        Construye la arquitectura completa llamando a _build_encoder() y _build_decoder().
        """
        self._build_encoder()
        self._build_decoder()

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

    def _train_autoencoder(self, x: torch.Tensor) -> None:
        """
        Entrenar el autoencoder.
        Args:
            x (torch.Tensor): Datos de entrada.
        """
        # Activar el modo entrenamiento del autoencoder
        self.train()
        train_ended_flag = False
        print("- Entrenando autoencoder...")

        # Inicializar optimizador (Adam)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        progress_bar = tqdm(range(self.epochs),
                            desc="Entrenamiento del autoencoder", unit="epoch")
        for epoch in progress_bar:
            epoch_loss = 0.0
            n_batches = 0

            # Generamos una serie de índices de manera aleatoria sin reemplazo
            # y los seleccionamos
            indices = torch.randperm(x.size(0), device=self.device)
            x_shuffled = x[indices]

            for i in range(0, x.size(0), self.batch_size):
                # Obtenemos el batch
                x_batch = x_shuffled[i:i + self.batch_size]

                # Entrenamos el autoencoder con el batch "x_batch"
                loss = self._train_batch(x_batch, optimizer)
                epoch_loss += loss
                n_batches += 1

            # Calculamos la pérdida promedio del epoch
            avg_loss = epoch_loss / n_batches

            # Mostrar pérdida promedio y epoch actual en la barra de progreso
            progress_bar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "epoch": f"{epoch+1}/{self.epochs}"
            })

            # Si la pérdida promedio es menor o igual al umbral establecido,
            # se finaliza el proceso de entrenamiento
            if avg_loss <= self.error_threshold:
                progress_bar.close()
                print(
                    f"- Entrenamiento detenido en epoch {epoch + 1}. Pérdida: {avg_loss:.4f}")
                train_ended_flag = True
                break

        if not train_ended_flag:
            progress_bar.close()
            print(f"- Entrenamiento completado. Pérdida final: {avg_loss:.4f}")

    def _train_batch(self, x_batch: torch.Tensor, optimizer: torch.optim.Optimizer):
        # Ponemos a cero los gradientes de los parámetros al comenzar a entrenar con un batch
        optimizer.zero_grad()

        # Añadimos ruido al batch
        x_batch_tilde = self._add_noise(x_batch)

        # Ejecutamos el forward de la red
        z, recon = self._forward(x_batch_tilde)

        # Obtenemos la pérdida (error de reconstrucción)
        loss = self.loss_fn(recon, x_batch)

        # Calculamos la pérdida total (p. ej. si hay regularización)
        total_loss = loss + self._compute_additional_loss(x_batch, z, recon)

        # Ejecutamos el backward de la red
        total_loss.backward()

        optimizer.step()
        return total_loss.item()

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

    def fit(self, data: np.ndarray) -> None:
        """
        Entrena el autoencoder utilizando un conjunto de datos.
        Args:
            data (np.ndarray): Matriz de N (número de patrones)
                               filas x D (número de características) columnas.
                               Establece el autoencoder como entrenado con el atributo
                               `self.trained`.
        """
        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        self._train_autoencoder(x)
        self.trained = True

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
