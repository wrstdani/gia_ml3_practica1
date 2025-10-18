from abc import ABC, abstractmethod

import numpy as np

import torch

class Autoencoder(torch.nn.Module):
    """
    
    """
    def __init__(self,
                 batch_size: int,
                 input_dim: int,
                 latent_dim: int = 32,
                 lr: float = 1e-3,
                 epochs: int = 100,
                 loss_fn: torch.nn.Module | None = None,
                 error_threshold: float = 0.0,
                 device: str = "cpu"):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, x.size(0), self.batch_size):
                x_batch = x[i:i + self.batch_size]
                loss = self._train_batch(x_batch, optimizer)

    
    def _train_batch(self, x_batch: torch.Tensor, optimizer: torch.optim.Optimizer):
        optimizer.zero_grad()
        output = self._forward(x_batch)
        loss = self.loss_fn(output, x_batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(self, data: np.ndarray):
        x = torch.tensor(data, dtype=torch.float32, device=self.device)
        self._train_autoencoder(x)
        self.trained = True

    def transform(self, data: np.ndarray) -> np.ndarray | None:
        if self.trained:
            ...
        
        return None
