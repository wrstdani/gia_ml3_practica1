"""
Utilidades para la ejecución del script principal
"""

import os
import time
import pandas as pd
import numpy as np

def read_csv_data(path2csv: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
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

def read_mnist(path: os.PathLike, return_labels: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    labels_numpy = data["label"].to_numpy(dtype=np.int32, copy=True)
    data.drop(labels=["label"], axis=1, inplace=True)
    data_numpy = data.to_numpy(dtype=np.float32, copy=True)
    if return_labels:
        return data_numpy, labels_numpy
    else:
        return data_numpy

def write_test_data(
        test_name: str,
        path: os.PathLike,
        trustworthiness_train: float,
        trustworthiness_test: float,
        elapsed_fit_train: float,
        elapsed_transform_train: float,
        elapsed_transform_test: float):
    ...

def create_subset(data: np.ndarray, num_samples: int, seed: int = 42):
    np.random.seed(seed)
    actual_size = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], size=actual_size, replace=False)
    return data[indices]
