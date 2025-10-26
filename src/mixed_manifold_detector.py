"""
Clase principal del sistema, compuesta por el Autoencoder y el algoritmo
de manifold learning
"""

import os
import argparse
import time
import pickle
import numpy as np
import sklearn
from sklearn.manifold import trustworthiness
from autoencoder import Autoencoder
from linear_autoencoder import LinearAutoencoder
from utils import load_csv_fashion_mnist, save_experiment, create_subset


class MixedManifoldDetector:
    """
    Clase que representa el sistema principal de la práctica.
    Se compone de un autoencoder y un algoritmo de manifold learning clásico.
    """
    def __init__(self,
                 input_dim: int | None = None,
                 autoencoder: Autoencoder | None = None,
                 manifold_alg: sklearn.base.TransformerMixin | None = None,
        ):
        """
        Constructor de la clase principal del sistema.
        Args:
            input_dim (int): Tamaño (número de características) de los datos de entrada
            autoencoder (Autoencoder): Autoencoder a entrenar para obtener representaciones latentes.
            manifold_alg (sklearn.base.TransformerMixin): Algoritmo clásico de manifold learning para
                                                          obtener representaciones 2D.
        """
        if input_dim is None and autoencoder is None:
            raise ValueError(
                "input_dim no puede ser NoneType si no se proporciona un autoencoder")
        elif input_dim is None:
            input_dim = autoencoder.input_dim

        if autoencoder is None:
            self.autoencoder = LinearAutoencoder(
                batch_size=32, input_dim=input_dim)
        else:
            self.autoencoder = autoencoder

        if manifold_alg is None:
            self.manifold_alg = sklearn.manifold.TSNE()
        else:
            self.manifold_alg = manifold_alg

        self.train_data = None
        self.train_embeddings = None
        self.train_manifold = None
        # NearestNeighbors para los patrones de entrenamiento que devuelve el vecino más cercano
        self.knn_train_data = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
        # NearestNeighbors para los embeddings
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

        self.autoencoder.fit(self.train_data)  # Entrenamos el autoencoder.
        # Obtenemos los embeddings con el autoencoder entrenado.
        self.train_embeddings = self.autoencoder.transform(self.train_data)
        print("- Ejecutando algoritmo de manifold learning sobre los embeddings...")
        self.train_manifold = self.manifold_alg.fit_transform(
            self.train_embeddings)  # Obtenemos la representación 2D de los embeddings.
        # Hacemos que la instancia de NearestNeighbors para los patrones de entrenamiento
        # los aprenda.
        self.knn_train_data.fit(self.train_data)
        # Hacemos que la instancia de NearestNeighbors para los embeddings los aprenda.
        self.knn_embeddings.fit(self.train_embeddings)

        return self.train_manifold.copy()

    def fit(self, train_data: np.ndarray):
        """
        Método que ejecuta fit_transform() sin devolver nada.
        Args:
            train_data (np.ndarray): Conjunto de datos de entrenamiento.
        """
        self.fit_transform(
            train_data)  # Ejecutamos fit_transform() sin devolver el resultado.

    def transform(self,
                  data: np.ndarray,
                  k: int = 5,
                  threshold: float = 1e-9
        ) -> np.ndarray:
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
            train_data_distances, train_data_indices = self.knn_train_data.kneighbors(
                [pattern])  # Obtener el vecino más cercano al nuevo patrón
            # Si la distancia es menor que el umbral
            if train_data_distances[0][0] < threshold:
                # Se determina que se trata del mismo patrón
                transformed.append(
                    self.train_manifold[train_data_indices[0][0]])
            else:  # Si son distintos patrones
                # Obtener los k vecinos más cercanos al embedding del nuevo patrón
                _, embedding_indices = self.knn_embeddings.kneighbors(
                    [embedding], n_neighbors=k)
                # Calcular embedding promedio con los vecinos obtenidos
                mean_embedding = np.mean(
                    self.train_manifold[embedding_indices[0]], axis=0)
                transformed.append(mean_embedding)

        return np.array(transformed, dtype=np.float32)

    def save(self, path: os.PathLike, name: str = "detector_base.pkl") -> None:
        """
        Permite guardar una instancia de la clase en un fichero .pkl.
        Args:
            path (os.PathLike): Directorio en que se almacenará el fichero .pkl.
            name (str): Nombre del fichero .pkl (por defecto es detector_base.pkl).
        """
        detector_path = os.path.join(path, name)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(detector_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: os.PathLike, name: str = "detector_base.pkl") -> "MixedManifoldDetector":
        """
        Permite cargar una instancia de la clase desde un fichero .pkl.
        Args:
            path (os.PathLike): Directorio en que se encuentra el fichero .pkl.
            name (str): Nombre del fichero .pkl (por defecto es detector_base.pkl).
        Output:
            Detector obtenido mediante la lectura del fichero .pkl
        """
        detector_path = os.path.join(path, name)
        with open(detector_path, "rb") as f:
            return pickle.load(f)


def main():
    """
    Código principal
    """
    parser = argparse.ArgumentParser(description="MixedManifoldDetector CLI")
    parser.add_argument(
        "path_train", help="Ruta al archivo CSV que contiene el conjunto de datos de train")
    parser.add_argument(
        "path_test", help="Ruta al archivo CSV que contiene el conjunto de datos de test")
    args = parser.parse_args()

    # Cargar datos
    data_train = load_csv_fashion_mnist(args.path_train)
    data_test = load_csv_fashion_mnist(args.path_test)
    num_samples_train = 300
    num_samples_test = 50
    data_train = create_subset(data_train, num_samples_train)
    data_test = create_subset(data_test, num_samples_test)

    # Creamos una instancia del detector
    input_dim = data_train.shape[1]

    elapsed_fit_train = None
    results_subpath = os.path.join("artifacts", "script")
    models_path = os.path.join(results_subpath, "models")
    csv_path = os.path.join(results_subpath, "script_output.csv")
    embeddings_path = os.path.join(results_subpath, "script_embeddings.pkl")
    if not os.path.exists(models_path):
        detector = MixedManifoldDetector(input_dim)
        # Entrenamos el detector
        start_fit_train = time.time()
        detector.fit(data_train)
        elapsed_fit_train = time.time() - start_fit_train
    else:
        detector = MixedManifoldDetector.load(os.path.join(models_path, "detector_base.pkl"))
    # Obtenemos la representación 2D de los datos de entrenamiento y test
    start_transform_train = time.time()
    output_train = detector.transform(data_train)
    elapsed_transform_train = time.time() - start_transform_train
    start_transform_test = time.time()
    output_test = detector.transform(data_test)
    elapsed_transform_test = time.time() - start_transform_test

    # Calcular valores de trustworthiness para train y test
    trustworthiness_train = trustworthiness(data_train, output_train)
    trustworthiness_test = trustworthiness(data_test, output_test)

    # Guardar datos
    save_experiment(
        csv_path,
        embeddings_path,
        "script_test",
        output_train,
        output_test,
        trustworthiness_train,
        trustworthiness_test,
        elapsed_fit_train,
        elapsed_transform_train,
        elapsed_transform_test
    )

    if elapsed_fit_train:
        detector.save(models_path)

if __name__ == "__main__":
    main()
