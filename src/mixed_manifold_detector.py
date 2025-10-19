from autoencoder import Autoencoder
from linear_autoencoder import LinearAutoencoder

import sys
import numpy as np
import pandas as pd
import sklearn
import torch

class MixedManifoldDetector:
    def __init__(self, input_dim: int, autoencoder: Autoencoder | None = None, manifold_alg: sklearn.base.TransformerMixin | None = None):
        """
        Constructor de la clase principal del sistema.
        Args:
            input_dim (int): Tamaño (número de características) de los datos de entrada
            autoencoder (Autoencoder): Autoencoder a entrenar para obtener representaciones latentes.
            manifold_alg (sklearn.base.TransformerMixin): Algoritmo clásico de manifold learning para
                                                          obtener representaciones 2D.
        """
        if autoencoder is None:
            self.autoencoder = LinearAutoencoder(128, input_dim)
        else:
            self.autoencoder = autoencoder

        if manifold_alg is None:
            self.manifold_alg = sklearn.manifold.TSNE()
        else:
            self.manifold_alg = manifold_alg

        self.train_data = None
        self.train_embeddings = None
        self.train_manifold = None
        self.knn_train_data = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
        self.knn_embeddings = sklearn.neighbors.NearestNeighbors()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Entrena el autoencoder con los datos, obtiene los embeddings generados por el autoencoder
        y les aplica el algoritmo de manifold learning elegido. También inicializa los objetos
        sklearn.neighbors.NearestNeighbors que se utilizarán para interpolación en transform().
        Args:
            data (np.ndarray): Conjunto de datos.
        Output:
            Representación 2D de la representación latente
        """
        self.train_data = data.copy()
        self.autoencoder.fit(data)
        self.train_embeddings = self.autoencoder.transform(data)
        self.train_manifold = self.manifold_alg.fit_transform(self.train_embeddings)
        self.knn_train_data.fit(self.train_data)
        self.knn_embeddings.fit(self.train_embeddings)

        return self.train_manifold.copy()

    def fit(self, train_data: np.ndarray):
        """
        Método que ejecuta fit_transform() sin devolver nada.
        Args:
            train_data (np.ndarray): Conjunto de datos de entrenamiento.
        """
        self.fit_transform(train_data)
    
    def transform(self, data: np.ndarray, k: int = 5, threshold: float = 1e-9) -> np.ndarray:
        """
        Método que, en caso de que los datos sean distintos a los de entrenamiento
        aplica interpolación con los k embeddings más cercanos al obtenido con
        autoencoder.transform(data), y, en caso de que los datos se hayan utilizado
        para entrenar, se devuelve el embedding generado previamente.
        Args:
            data (np.ndarray): Datos a transformar.
            k (int): Número de vecinos a tener en cuenta para la interpolación.
            threshold (float): Umbral que determina si dos patrones son el mismo.
        Output:
            Representación 2D de la representación latente
        """
        transformed = []
        new_embeddings = self.autoencoder.transform(data)
        
        for pattern, embedding in zip(data, new_embeddings):
            train_data_distances, train_data_indices = self.knn_train_data.kneighbors([pattern])
            if train_data_distances[0][0] < threshold:
                transformed.append(self.train_manifold[train_data_indices[0][0]])
            else:
                _, embedding_indices = self.knn_embeddings.kneighbors([embedding], n_neighbors=k)
                mean_embedding = np.mean(self.train_manifold[embedding_indices[0]], axis=0)
                transformed.append(mean_embedding)

        return np.array(transformed, dtype=np.float32)


# Código principal
def main():
    if len(sys.argv) < 2:
        print("Uso: python mixed_manifold_detector.py <archivo.csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    print(f"- Cargando conjunto de datos desde {csv_path}...")
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        y_train = df["label"].to_numpy()  # Labels
        df.drop(labels=["label"], axis=1, inplace=True)  # Eliminamos la columna label del df
    data = df.to_numpy(dtype=np.float32)  # Convertimos los datos en un np.ndarray

    detector = MixedManifoldDetector(
        input_dim=data.shape[1]
    )  # Inicializamos el detector (configuración por defecto)
    output = detector.fit_transform(data)
    print(output)


if __name__ == "__main__":
    main()
