"""
Utilidades para la ejecución del script principal y de los tests
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


def load_image_dataset(
    name: str,
    return_labels: bool = False,
    return_test: bool = True,
    normalize: bool = True,
    seed: int = 42
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | \
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
    """
    Carga un dataset de imágenes. Los datasets válidos son mnist_784, 
    Fashion-MNIST y cifar_10_small.
    Args:
        name (str): Nombre del dataset a cargar.
        return_labels (bool): Determina si se devuelven las etiquetas o no.
        return_test (bool): Determina si se devuelve el conjunto de test o sólo el de train.
        normalize (bool): Determina si se normalizan los datos o no.
        seed (int): Semilla para recrear experimentos.
    """
    valid_datasets = set(("mnist_784", "Fashion-MNIST", "cifar_10_small"))
    if name in valid_datasets:
        loaded_data = fetch_openml(name)
        data, labels = loaded_data["data"].to_numpy(
            dtype=np.float32), loaded_data["target"].to_numpy(dtype=np.int32)
        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, test_size=0.2, random_state=seed)
        if normalize:
            data_train /= 255.0
            data_test /= 255.0
        return (
            ((data_train, labels_train), (data_test, labels_test)
             ) if return_labels else (data_train, data_test)
        ) if return_test else (
            (data_train, labels_train) if return_labels else data_train
        )

    else:
        raise ValueError(
            f"El dataset {name} no está soportado. Utiliza uno de los siguientes: {valid_datasets}")


def load_glass_identification_dataset(
    return_labels: bool = False,
    return_test: bool = True,
    normalize: bool = True,
    seed: int = 42
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | \
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Carga el dataset Glass Identification (único dataset para pruebas cuyos datos no son imágenes).
    Args:
        return_labels (bool): Determina si se devuelven las etiquetas o no.
        return_test (bool): Determina si se devuelve el conjunto de test o sólo el de train.
        normalize (bool): Determina si se normalizan o no los datos.
        seed (int): Semilla para recrear experimentos.
    """
    loaded_data = fetch_openml("Glass-Classification")
    data, labels = loaded_data["data"].iloc[:, 0:8].to_numpy(
        dtype=np.float32), loaded_data["data"].iloc[:, 9].to_numpy(dtype=np.int32)
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=seed)
    if normalize:
        std = StandardScaler()
        data_train = std.fit_transform(data_train)
        data_test = std.transform(data_test)
    if return_test:
        return ((data_train, labels_train), (data_test, labels_test)) \
            if return_labels else (data_train, data_test)
    else:
        return (data_train, labels_train) if return_labels else data_train


def load_csv_fashion_mnist(
    path: os.PathLike,
    return_labels: bool = False,
    normalize: bool = True
):
    """
    Carga un dataset de Fashion MNIST en formato CSV.
    Args:
        path (os.PathLike): Ruta al fichero CSV.
        return_labels (bool): Determina si se devuelven las etiquetas o no.
        normalize (bool): Determina si se normaliza el conjunto de datos.
    """
    df = pd.read_csv(path)
    labels = df["label"].to_numpy(dtype=np.int32, copy=True)
    df.drop(labels=["label"], axis=1, inplace=True)
    data = df.to_numpy(dtype=np.float32, copy=True)
    if normalize:
        data /= 255.0
    return ((data, labels) if return_labels else data)


def create_subset(
    data: np.ndarray,
    num_samples: int,
    labels: np.ndarray | None = None,
    seed: int = 42
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Devuelve un subconjunto de datos del proporcionado en los argumentos.
    Dicho subconjunto se elige de manera aleatoria.
    Args:
        data (np.ndarray): Conjunto de datos del cuál se quiere obtener el subconjunto.
        num_samples (int): Número de patrones que tendrá el subconjunto.
        labels (np.ndarray): Etiquetas del conjunto de datos.
        seed (int): Semilla para recrear experimentos.
    """
    np.random.seed(seed)
    actual_size = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], size=actual_size, replace=False)
    return ((data[indices, :], labels[indices]) if (labels is not None) else data[indices, :])


def save_experiment(
    csv_path: os.PathLike,
    embeddings_path: os.PathLike,
    labels_path: os.PathLike,
    experiment_name: str,
    embedding_train: np.ndarray,
    labels_train: np.ndarray,
    embedding_test: np.ndarray,
    labels_test: np.ndarray,
    trustworthiness_train: float,
    trustworthiness_test: float,
    elapsed_fit_train: float | None,
    elapsed_transform_train: float,
    elapsed_transform_test: float
) -> None:
    """
    Permite guardar lás métricas generadas por un experimento.
    Args:
        csv_path (os.PathLike): Ruta al fichero CSV en que se guardarán los datos.
        embeddings_path (os.PathLike): Ruta al fichero .pkl en que se guardarán los embeddings.
        experiment_name (str): Nombre del experimento.
        embedding_train (np.ndarray): Embedding del conjunto de entrenamiento.
        embedding_test (np.ndarray): Embedding del conjunto de test.
        trustworthiness_train (float): Trustworthiness del embedding de entrenamiento respecto
                                       a los datos de entrenamiento.
        trustworthiness_test (float): Trustworthiness del embedding de test respecto a los datos
                                      de test.
        elapsed_fit_train (float): Tiempo (en segundos) que el sistema tarda en ejecutar fit()
                                   sobre los datos de entrenamiento.
        elapsed_transform_train (float): Tiempo (en segundos) que el sistema tarda en ejecutar
                                         transform() sobre los datos de entrenamiento.
        elapsed_transform_test (float): Tiempo (en segundos) que el sistema tarda en ejecutar
                                        transform() sobre los datos de test.
    """
    path_dir_csv = os.path.dirname(csv_path)
    path_dir_embeddings = os.path.dirname(embeddings_path)
    os.makedirs(path_dir_csv, exist_ok=True)
    os.makedirs(path_dir_embeddings, exist_ok=True)
    test_dict = {
        "test_id": [experiment_name],
        "trustworthiness_train": [trustworthiness_train],
        "trustworthiness_test": [trustworthiness_test],
        "elapsed_fit_train": [elapsed_fit_train],
        "elapsed_transform_train": [elapsed_transform_train],
        "elapsed_transform_test": [elapsed_transform_test]
    }
    embeddings_dict = {
        "embedding_train": embedding_train,
        "embedding_test": embedding_test
    }
    labels_dict = {
        "labels_train": labels_train,
        "labels_test": labels_test
    }

    df = pd.DataFrame(test_dict)
    df.set_index("test_id")

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', index=False)

    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings_dict, f)

    with open(labels_path, "wb") as f:
        pickle.dump(labels_dict, f)
