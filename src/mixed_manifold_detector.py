"""
Clase principal del sistema, compuesta por el Autoencoder y el algoritmo
de manifold learning
"""

import sys
import numpy as np
import sklearn
import torch
from autoencoder import Autoencoder
from linear_autoencoder import LinearAutoencoder
from linear_sparse_autoencoder import LinearSparseAutoencoder
from denoising_sparse_autoencoder import DenoisingSparseAutoencoder
from utils import read_csv_data

class MixedManifoldDetector:
    def __init__(self,
                 input_dim: int | None = None,
                 autoencoder: Autoencoder | None = None,
                 manifold_alg: sklearn.base.TransformerMixin | None = None,
                 scaler: sklearn.base.TransformerMixin | None = None,
                 scale_flag: bool = True):
        """
        Constructor de la clase principal del sistema.
        Args:
            input_dim (int): Tamaño (número de características) de los datos de entrada
            autoencoder (Autoencoder): Autoencoder a entrenar para obtener representaciones latentes.
            manifold_alg (sklearn.base.TransformerMixin): Algoritmo clásico de manifold learning para
                                                          obtener representaciones 2D.
            scaler (sklearn.base.TransformerMixin): Transformador para escalar los datos. Por defecto,
                                                    se utiliza StandardScaler si scale_flag=True
            scale_flag (bool): Si es True, aplica el escalado a los datos. Si no, se utilizan los datos originales.
        """
        if input_dim is None and autoencoder is None:
            raise ValueError("input_dim no puede ser NoneType si no se proporciona un autoencoder")
        elif input_dim is None:
            input_dim = autoencoder.input_dim

        if autoencoder is None:
            self.autoencoder = LinearAutoencoder(batch_size=32, input_dim=input_dim)
        else:
            self.autoencoder = autoencoder

        if manifold_alg is None:
            self.manifold_alg = sklearn.manifold.TSNE()
        else:
            self.manifold_alg = manifold_alg

        self.scale_flag = scale_flag
        if scaler is None and self.scale_flag:
            self.scaler = sklearn.preprocessing.StandardScaler()
        else:
            self.scaler = scaler

        self.train_data = None
        self.train_embeddings = None
        self.train_manifold = None
        self.knn_train_data = sklearn.neighbors.NearestNeighbors(n_neighbors=1)  # NearestNeighbors para los patrones de entrenamiento que devuelve el vecino más cercano
        self.knn_embeddings = sklearn.neighbors.NearestNeighbors()  # NearestNeighbors para los embeddings

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

        if self.scale_flag:
            data_scaled = self.scaler.fit_transform(self.train_data)  # Aplicamos transformación a los datos
        else:
            data_scaled = data
        self.autoencoder.fit(data_scaled)  # Entrenamos el autoencoder
        self.train_embeddings = self.autoencoder.transform(data_scaled)  # Obtenemos los embeddings con el autoencoder entrenado
        print("- Ejecutando algoritmo de manifold learning sobre los embeddings...")
        self.train_manifold = self.manifold_alg.fit_transform(self.train_embeddings)  # Obtenemos la representación 2D de los embeddings
        self.knn_train_data.fit(self.train_data)  # Hacemos que la instancia de NearestNeighbors para los patrones de entrenamiento los aprenda
        self.knn_embeddings.fit(self.train_embeddings)  # Hacemos que la instancia de NearestNeighbors para los embeddings los aprenda

        return self.train_manifold.copy()

    def fit(self, train_data: np.ndarray):
        """
        Método que ejecuta fit_transform() sin devolver nada.
        Args:
            train_data (np.ndarray): Conjunto de datos de entrenamiento.
        """
        self.fit_transform(train_data)  # Ejecutamos fit_transform() sin devolver el resultado
    
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
        if self.scale_flag:
            data_scaled = self.scaler.transform(data)
        else:
            data_scaled = data

        transformed = []
        new_embeddings = self.autoencoder.transform(data_scaled)
        
        for pattern, embedding in zip(data, new_embeddings):
            train_data_distances, train_data_indices = self.knn_train_data.kneighbors([pattern])  # Obtener el vecino más cercano al nuevo patrón
            if train_data_distances[0][0] < threshold:  # Si la distancia es menor que el umbral
                transformed.append(self.train_manifold[train_data_indices[0][0]])  # Se determina que se trata del mismo patrón
            else:  # Si son distintos patrones
                _, embedding_indices = self.knn_embeddings.kneighbors([embedding], n_neighbors=k)  # Obtener los k vecinos más cercanos al embedding del nuevo patrón
                mean_embedding = np.mean(self.train_manifold[embedding_indices[0]], axis=0)  # Calcular embedding promedio con los vecinos obtenidos
                transformed.append(mean_embedding)

        return np.array(transformed, dtype=np.float32)


# Código principal
def main():
    if len(sys.argv) > 2:
        print("- Uso:")
        print("$ python mixed_manifold_detector.py <data_train.csv> <data_test.csv>")
    
    data_train_path = sys.argv[1]
    print(f"- Cargando conjunto de datos desde {csv_path}...")
    data, labels = read_csv_data(csv_path)

    input_dim = data.shape[1]
    autoencoder = LinearSparseAutoencoder(
        batch_size=128,
        input_dim=input_dim,
        latent_dim=32,
        lr=1e-3,
        epochs=100,
        loss_fn=None,  # Esto hace que utilice la función de pérdida por defecto (torch.nn.MSELoss)
        error_threshold=0.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lambda_val=1e-3
    )
    manifold_alg = sklearn.manifold.LocallyLinearEmbedding(n_components=2)

    detector = MixedManifoldDetector(
        input_dim=input_dim,
        autoencoder=autoencoder,
        manifold_alg=manifold_alg,
        scaler=None,  # Esto hace que se utilice el scaler por defecto (sklearn.preprocessing.StandardScaler)
        scale_flag=True  # Activa el funcionamiento del scaler
    )

    output = detector.fit_transform(data)

    print("- Información del output:")
    print(f"    - Dimensiones del output: {output.shape}")
    print(f"    - dtype del output: {output.dtype}")
    print(f"    - Trustworthiness respecto al conjunto original: {sklearn.manifold.trustworthiness(data, output)}")
    

if __name__ == "__main__":
    main()
