"""
Utilidades para la ejecución del script principal
"""

import os
import time
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_mnist_csv_file(path2csv: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Función auxiliar para leer datos en formato CSV con la estructura
    que se especifica en el enunciado de la práctica y obtener
    una matriz de NumPy que los contiene. El dtype de la matriz
    devuelta será np.float32 y el del vector de labels será np.int32.
    Args:
        path2csv (os.PathLike): Ruta al fichero CSV
    Output:
        Matriz de NumPy con los datos leídos y vector de NumPy
        con las labels.
    """
    data = pd.read_csv(path2csv)
    labels_numpy = data["label"].to_numpy(dtype=np.int32, copy=True)
    data.drop(labels=["label"], axis=1, inplace=True)
    data_numpy = data.to_numpy(dtype=np.float32, copy=True)
    return data_numpy, labels_numpy


def read_cifar10_batch(path2batch: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    with open(path2batch, "rb") as f:
        loaded = pickle.load(f, encoding="bytes")
    return np.float32(loaded[b"data"]), np.int32(loaded[b"labels"])


def read_glass_identification_csv_file(path2csv: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path2csv)
    labels = df.iloc[:, 10].to_numpy(dtype=np.int32, copy=True)
    df.drop(labels=[df.columns[0], df.columns[10]], axis=1, inplace=True)
    data = df.to_numpy(dtype=np.float32, copy=True)
    return data, labels


def read_mnist(
    path_train: os.PathLike,
    path_test: os.Pathlike | None = None,
    return_labels: bool = False,
    normalize: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    data_train, labels_train = read_mnist_csv_file(path_train)
    if normalize:
        data_train /= 255.0
    if path_test is not None:
        data_test, labels_test = read_mnist_csv_file(path_test)
        if normalize:
            data_test /= 255.0
        return ((data_train, labels_train), (data_test, labels_test)) if return_labels else (data_train, data_test)
    return (data_train, labels_train) if return_labels else data_train


def read_cifar10(
    path_to_batches: os.PathLike,
    return_labels: bool = False,
    return_test: bool = True,
    normalize: bool = True
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Lee los datos del dataset CIFAR-10. Este conjunto de datos está compuesto
    por 5 batches de entrenamiento (50.000 imágenes) y 1 batch de test (10.000 imágenes).
    """
    batches_paths = [os.path.join(path_to_batches, f) for f in os.listdir(
        path_to_batches) if (f.startswith("data_batch_") or (return_test and f == "test_batch"))]
    train_batches = []
    for b in batches_paths:
        if b.find("test_batch"):
            data_test, labels_test = read_cifar10_batch(b)
            if normalize:
                data_test /= 255.0
        else:
            train_batches.append((read_cifar10_batch(b)))
    data_train = np.vstack([batch[0]
                           for batch in train_batches], dtype=np.float32)
    if normalize:
        data_train /= 255.0
    labels_train = np.vstack([batch[1]
                             for batch in train_batches], dtype=np.int32)
    return (
        ((data_train, labels_train), (data_test, labels_test)
         ) if return_labels else (data_train, data_test)
    ) if return_test else (
        (data_train, labels_train) if return_labels else data_train
    )


def read_glass_identification(
    path: os.PathLike,
    return_labels: bool = False,
    return_test: bool = True,
    normalize: bool = True,
    test_size: float = 0.2,
    seed: int = 42
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    data, labels = read_glass_identification_csv_file(path)
    if normalize:
        std = StandardScaler()
        data = std.fit_transform(data)
    if return_test:
        data_train, data_test, labels_train, labels_test = train_test_split(
            data, labels, test_size=test_size, random_state=seed
        )
        return ((data_train, labels_train), (data_test, labels_test)) if return_labels else (data_train, data_test)
    else:
        return (data, labels) if return_labels else data


def write_test_data(
    test_name: str,
    path: os.PathLike,
    trustworthiness_train: float,
    trustworthiness_test: float,
    elapsed_fit_train: float,
    elapsed_transform_train: float,
    elapsed_transform_test: float
) -> None:
    test_dict = {
        "test_id": [test_name],
        "trustworthiness_train": [trustworthiness_train],
        "trustworthiness_test": [trustworthiness_test],
        "elapsed_fit_train": [elapsed_fit_train],
        "elapsed_transform_train": [elapsed_transform_train],
        "elapsed_transform_test": [elapsed_transform_test]
    }

    df = pd.DataFrame(test_dict)
    df.set_index("test_id")

    if os.path.exists(path):
        df.to_csv(path, mode='a', header=False, index=False)
    else:
        df.to_csv(path, mode='w', index=False)


def create_subset(
    data: np.ndarray,
    num_samples: int,
    seed: int = 42
) -> np.ndarray:
    np.random.seed(seed)
    actual_size = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], size=actual_size, replace=False)
    return data[indices]
