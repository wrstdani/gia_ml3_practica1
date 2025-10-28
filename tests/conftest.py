"""
Configuración y constantes para testing
"""

import os
import sys
import pytest
import numpy as np
import sklearn
from sklearn.manifold import TSNE, LocallyLinearEmbedding
import itertools

# Incluir la raíz del proyecto en el path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(root_path, "src")
sys.path.insert(0, src_path)

# Constantes
RANDOM_SEED = 42
AUTOENCODER_TYPES = [
    "LinearAutoencoder",
    "LinearSparseAutoencoder",
    "DenoisingSparseAutoencoder",
    "VariationalAutoencoder"
]
MANIFOLD_ALGORITHMS = [
    "tsne",
    "lle"
]
AUTOENCODER_MANIFOLD_COMBINATIONS = list(
    itertools.product(AUTOENCODER_TYPES, MANIFOLD_ALGORITHMS))
DATASET_NAMES = ["mnist_784", "Fashion-MNIST",
                 "cifar_10_small", "Glass-Classification"]
DATASET_FIXTURES = tuple(zip(DATASET_NAMES, ("mnist_data",
                                             "fashion_mnist_data", "cifar10_data", "glass_data")))

# Fixtures para cargar los conjuntos de datos


@pytest.fixture(scope="session")
def mnist_data():
    from utils import load_image_dataset
    return load_image_dataset("mnist_784", True, seed=RANDOM_SEED)


@pytest.fixture(scope="session")
def fashion_mnist_data():
    from utils import load_image_dataset
    return load_image_dataset(
        "Fashion-MNIST", True, seed=RANDOM_SEED)


@pytest.fixture(scope="session")
def cifar10_data():
    from utils import load_image_dataset
    return load_image_dataset(
        "cifar_10_small", True, seed=RANDOM_SEED)


@pytest.fixture(scope="session")
def glass_data():
    from utils import load_glass_identification_dataset
    return load_glass_identification_dataset(True, seed=RANDOM_SEED)

# Fixtures de rutas


@pytest.fixture(scope="session")
def data_path():
    """
    Ruta al directorio de datos
    """
    return os.path.abspath(os.path.join(root_path, "data"))


@pytest.fixture(scope="session")
def results_path():
    """
    Ruta al directorio de resultados
    """
    path = os.path.abspath(os.path.join(root_path, "artifacts"))
    os.makedirs(path, exist_ok=True)
    return path

# Fixtures de datos sintéticos


@pytest.fixture
def sample_data():
    """
    Genera datos sintéticos
    """
    np.random.seed(42)
    data = np.random.randn(100, 50).astype(np.float32)
    return data

# Fixtures de autoencoders


@pytest.fixture
def autoencoder_factory():
    """
    Factory para crear autoencoders para testing
    """
    def _create_autoencoder(autoencoder_type: str, input_dim: int = 784, **kwargs):
        default_params = {
            "batch_size": 32,
            "input_dim": input_dim,
            "latent_dim": 32,
            "lr": 1e-3,
            "epochs": 100,
            "device": "cpu"
        }
        params = {**default_params, **kwargs}

        if autoencoder_type == "LinearAutoencoder":
            from linear_autoencoder import LinearAutoencoder
            return LinearAutoencoder(**params)

        elif autoencoder_type == "LinearSparseAutoencoder":
            from linear_sparse_autoencoder import LinearSparseAutoencoder
            return LinearSparseAutoencoder(
                **params,
                lambda_val=kwargs.get("lambda_val", 1e-3)
            )

        elif autoencoder_type == "DenoisingSparseAutoencoder":
            from denoising_sparse_autoencoder import DenoisingSparseAutoencoder
            return DenoisingSparseAutoencoder(
                **params,
                lambda_val=kwargs.get("lambda_val", 1e-3),
                noise_factor=kwargs.get("noise_factor", 0.3)
            )

        elif autoencoder_type == "VariationalAutoencoder":
            from variational_autoencoder import VariationalAutoencoder
            return VariationalAutoencoder(**params)

        else:
            raise ValueError(
                f"Tipo de autoencoder desconocido: {autoencoder_type}")

    return _create_autoencoder

# Fixtures del detector


@pytest.fixture
def mixed_manifold_detector_factory():
    """
    Factory para crear instancias de MixedManifoldDetector
    """
    def _create_mixed_manifold_detector(input_dim: int = 784, manifold_alg: str = "tsne", **kwargs):
        default_params = {
            "input_dim": input_dim,
            "autoencoder": None,
            "manifold_alg": None
        }
        params = {**default_params, **kwargs}

        if manifold_alg == "tsne":
            params["manifold_alg"] = TSNE()
        elif manifold_alg == "lle":
            params["manifold_alg"] = LocallyLinearEmbedding()
        else:
            raise ValueError(
                f"Tipo de algoritmo de manifold desconocido: {manifold_alg}")

        from mixed_manifold_detector import MixedManifoldDetector
        return MixedManifoldDetector(**params)

    return _create_mixed_manifold_detector
