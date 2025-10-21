import os
import pytest
import time
import itertools
from sklearn.manifold import trustworthiness
from tests.conftest import AUTOENCODER_MANIFOLD_COMBINATIONS
from src.utils import write_test_data, create_subset


@pytest.fixture(scope="module")
def fashion_mnist_data(data_path):
    import os
    from src.utils import read_mnist

    path_train = os.path.join(
        data_path, "fashion-mnist", "fashion-mnist_train.csv")
    path_test = os.path.join(data_path, "fashion-mnist",
                             "fashion-mnist_test.csv")
    data_train, data_test = read_mnist(path_train, path_test)

    return data_train, data_test


@pytest.mark.parametrize("autoencoder_type,manifold_alg", AUTOENCODER_MANIFOLD_COMBINATIONS)
@pytest.mark.parametrize("size_label,train_subset_size,test_subset_size", [
    ("very_small", 100, 20),
    ("small", 300, 50),
    ("medium", 500, 100),
    ("medium_large", 1000, 200),
    ("large", 5000, 1000),
    ("very_large", 10000, 2000)
])
def test_fashion_mnist(
    fashion_mnist_data,
    autoencoder_factory,
    mixed_manifold_detector_factory,
    size_label,
    train_subset_size,
    test_subset_size,
    autoencoder_type,
    manifold_alg,
    results_path
):
    data_train, data_test = fashion_mnist_data
    data_train_small = create_subset(data_train, train_subset_size, 42)
    data_test_small = create_subset(data_test, test_subset_size, 42)

    autoencoder = autoencoder_factory(
        autoencoder_type, data_train_small.shape[1])
    detector = mixed_manifold_detector_factory(
        data_train_small.shape[1], manifold_alg, autoencoder=autoencoder)

    start_fit_train = time.time()
    detector.fit(data_train_small)
    elapsed_fit_train = time.time() - start_fit_train

    start_transform_train = time.time()
    output_train = detector.transform(data_train_small)
    elapsed_transform_train = time.time() - start_transform_train
    trustworthiness_train = trustworthiness(data_train_small, output_train)

    start_transform_test = time.time()
    output_test = detector.transform(data_test_small)
    elapsed_transform_test = time.time() - start_transform_test
    trustworthiness_test = trustworthiness(data_test_small, output_test)

    test_name = "fashion-mnist_" + \
        f"{size_label}".lower() + f"{autoencoder_type}".lower() + \
        f"_{manifold_alg}".lower()

    write_test_data(
        test_name,
        os.path.join(results_path, "tests_fashion-mnist_output.csv"),
        trustworthiness_train,
        trustworthiness_test,
        elapsed_fit_train,
        elapsed_transform_train,
        elapsed_transform_test
    )
