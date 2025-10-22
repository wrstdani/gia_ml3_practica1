import os
import pytest
import time
from sklearn.manifold import trustworthiness
from tests.conftest import AUTOENCODER_MANIFOLD_COMBINATIONS, RANDOM_SEED, DATASETS
from src.utils import save_experiment, create_subset

@pytest.fixture(scope="session")
def mnist_data():
    from src.utils import load_image_dataset
    data_train, data_test = load_image_dataset("Fashion-MNIST", seed=RANDOM_SEED)
    return data_train, data_test

@pytest.fixture(scope="session")
def fashion_mnist_data():
    from src.utils import load_image_dataset
    data_train, data_test = load_image_dataset("mnist_784", seed=RANDOM_SEED)
    return data_train, data_test

@pytest.fixture(scope="session")
def cifar10_data():
    from src.utils import load_image_dataset
    data_train, data_test = load_image_dataset("cifar_10_small", seed=RANDOM_SEED)
    return data_train, data_test

@pytest.fixture(scope="session")
def glass_data():
    from src.utils import load_glass_identification_dataset
    data_train, data_test = load_glass_identification_dataset(seed=RANDOM_SEED)
    return data_train, data_test


DATASET_FIXTURES = {
    "mnist_784": "mnist_data",
    "Fashion-MNIST": "fashion_mnist_data",
    "cifar_10_small": "cifar10_data",
    "Glass-Classification": "glass_data"
}


@pytest.mark.parametrize("dataset_name", DATASETS)
@pytest.mark.parametrize("autoencoder_type,manifold_alg", AUTOENCODER_MANIFOLD_COMBINATIONS)
@pytest.mark.parametrize("size_label,train_subset_size,test_subset_size", [
    ("small", 300, 50),
    ("medium_large", 1000, 200),
    ("large", 5000, 1000),
])
def test_metrics_subsets(
    dataset_name,
    autoencoder_factory,
    mixed_manifold_detector_factory,
    size_label,
    train_subset_size,
    test_subset_size,
    autoencoder_type,
    manifold_alg,
    results_path,
    request
):
    fixture_name = DATASET_FIXTURES[dataset_name]
    data_train, data_test = request.getfixturevalue(fixture_name)
    data_train_small = create_subset(data_train, num_samples=train_subset_size, seed=RANDOM_SEED)
    data_test_small = create_subset(data_test, num_samples=test_subset_size, seed=RANDOM_SEED)

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

    results_subpath = os.path.join(results_path, f"{dataset_name}")
    if not os.path.exists(results_subpath):
        os.makedirs(results_subpath)
    test_name = f"{dataset_name}_{size_label}_{autoencoder_type}_{manifold_alg}".lower()
    csv_filename = f"tests_{dataset_name}_output.csv"
    embeddings_filename = f"{dataset_name}_{size_label}_{autoencoder_type}_{manifold_alg}.pkl".lower()

    save_experiment(
        os.path.join(results_subpath, csv_filename),
        os.path.join(results_subpath, embeddings_filename),
        test_name,
        output_train,
        output_test,
        trustworthiness_train,
        trustworthiness_test,
        elapsed_fit_train,
        elapsed_transform_train,
        elapsed_transform_test
    )
