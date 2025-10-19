from abc import ABC, abstractmethod

import numpy as np

import torch

class Autoencoder(torch.nn.Module, ABC):
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
        Constructor base de la interfaz Autoencoder.
        Args:
            batch_size (int): Tamaño de los batches para entrenar el autoencoder.
            input_dim (int): Tamaño de los datos de entrada (número de características/dimensiones).
            latent_dim (int): Tamaño de la capa de embedding del autoencoder (la salida del encoder).
            lr (float): Learning rate para descenso del gradiente en el entrenamiento del autoencoder.
            epochs (int): Número de epochs de entrenamiento del autoencoder.
            loss_fn (torch.nn.Module): Función de pérdida para el entrenamiento del autoencoder.
            error_threshold (float): Umbral que determina si se continúa entrenando el autoencoder en cada epoch.
            device (str): Dispositivo en que PyTorch entrenará el autoencoder.
        """
        super().__init__()

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn if loss_fn is not None else torch.nn.MSELoss()
        self.error_threshold = error_threshold
        self.device = device
        self.trained = False  # Para prevenir utilizar transform() sin entrenar
        self.encoder = None
        self.decoder = None
        self._build_architecture()
        self.to(self.device)

    @abstractmethod
    def _build_encoder(self):
        pass

    @abstractmethod
    def _build_decoder(self):
        pass

    def _build_architecture(self):
        self._build_encoder()
        self._build_decoder()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
    
    def _train_autoencoder(self, x: torch.Tensor):
        self.train()
        print("- Entrenando autoencoder...")
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):

            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(x.size(0), device=self.device)
            x_shuffled = x[indices]

            for i in range(0, x.size(0), self.batch_size):
                x_batch = x_shuffled[i:i + self.batch_size]
                loss = self._train_batch(x_batch, optimizer)
                epoch_loss += loss
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            print(f"    - Pérdida promedio en epoch {epoch + 1} = {avg_loss:.4f}")
            if avg_loss <= self.error_threshold:
                print(f"- Entrenamiento detenido en epoch {epoch + 1}. Pérdida: {avg_loss:.4f}")

    
    def _train_batch(self, x_batch: torch.Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        output = self._forward(x_batch)
        loss = self.loss_fn(output, x_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(self, data: np.ndarray):
        """
        Entrena el autoencoder utilizando un conjunto de datos.
        Args:
            data (np.ndarray): Matriz de N (número de patrones) filas x D (número de características) columnas.
                            Establece el autoencoder como entrenado con el atributo `self.trained`.
        """
        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        self._train_autoencoder(x)
        self.trained = True

    
    def transform(self, data: np.ndarray) -> np.ndarray | None:
        """
        Requiere que el autoencoder haya sido entrenado previamente con `fit()`. Se utiliza para obtener
        la representación latente (salida del encoder) del autoencoder de un conjunto de datos.
        Args:
            data (np.ndarray): Datos de los cuales se desea obtener el embedding.
        Output:
            np.ndarray
            None
        """
        if not self.trained:
            return None
        
        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32, device=self.device)
            embeddings = self.encoder(data_tensor)
        return embeddings.cpu().numpy()
