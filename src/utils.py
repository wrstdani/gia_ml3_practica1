"""
Utilidades para la ejecución del script principal
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
# from mixed_manifold_detector import MixedManifoldDetector


def load_image_dataset(
    name: str,
    return_labels: bool = False,
    return_test: bool = True,
    normalize: bool = True,
    seed: int = 42
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
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
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
        return ((data_train, labels_train), (data_test, labels_test)) if return_labels else (data_train, data_test)
    else:
        return (data_train, labels_train) if return_labels else data_train


def load_csv_fashion_mnist(
    path: os.PathLike,
    return_labels: bool = False,
    normalize: bool = True
):
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
    np.random.seed(seed)
    actual_size = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], size=actual_size, replace=False)
    return ((data[indices, :], labels[indices]) if labels else data[indices, :])


def save_experiment(
    csv_path: os.PathLike,
    embeddings_path: os.PathLike,
    test_name: str,
    embedding_train: np.ndarray,
    embedding_test: np.ndarray,
    trustworthiness_train: float,
    trustworthiness_test: float,
    elapsed_fit_train: float | None,
    elapsed_transform_train: float,
    elapsed_transform_test: float
) -> None:
    path_dir_csv = os.path.dirname(csv_path)
    path_dir_embeddings = os.path.dirname(embeddings_path)
    if not os.path.exists(path_dir_csv):
        os.makedirs(path_dir_csv)
    if not os.path.exists(path_dir_embeddings):
        os.makedirs(path_dir_embeddings)
    test_dict = {
        "test_id": [test_name],
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

    df = pd.DataFrame(test_dict)
    df.set_index("test_id")

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', index=False)

    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings_dict, f)
