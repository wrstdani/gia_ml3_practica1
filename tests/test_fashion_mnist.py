import os
import pytest
import time
import itertools
from sklearn.manifold import trustworthiness
from src.utils import write_test_data, create_subset

autoencoders = ["LinearAutoencoder", "LinearSparseAutoencoder", "DenoisingSparseAutoencoder"]
manifolds = ["lle", "tsne"]
combinations = list(itertools.product(autoencoders, manifolds))

@pytest.fixture(scope="module")
def fashion_mnist_data(data_path):
    import os
    from src.utils import read_mnist

    data_train = read_mnist(os.path.join(data_path, "fashion-mnist", "fashion-mnist_train.csv"))
    data_test = read_mnist(os.path.join(data_path, "fashion-mnist", "fashion-mnist_test.csv"))

    return data_train, data_test

@pytest.mark.parametrize("autoencoder_type,manifold_alg", combinations)
@pytest.mark.parametrize("train_subset_size,test_subset_size", [(300, 50)])
def test_fashion_mnist(
    fashion_mnist_data,
    autoencoder_factory,
    mixed_manifold_detector_factory,
    train_subset_size,
    test_subset_size,
    autoencoder_type,
    manifold_alg,
    results_path
    ):
    data_train, data_test = fashion_mnist_data
    data_train_small = create_subset(data_train, train_subset_size, 42)
    data_test_small = create_subset(data_test, test_subset_size, 42)

    autoencoder = autoencoder_factory(autoencoder_type, data_train_small.shape[1])
    detector = mixed_manifold_detector_factory(data_train_small.shape[1], manifold_alg, autoencoder=autoencoder)

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

    test_name = f"{autoencoder_type}".lower() + f"{manifold_alg}".lower()

    write_test_data(
        os.path.join(results_path, "fashion-mnist_tests.csv"),
        test_name,
        trustworthiness_train,
        trustworthiness_test,
        elapsed_fit_train,
        elapsed_transform_train,
        elapsed_transform_test
    )
