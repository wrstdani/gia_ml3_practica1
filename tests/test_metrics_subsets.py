"""
Testing para recoger métricas y representaciones de subconjuntos de datos
"""

import os
import pytest
import time
from sklearn.manifold import trustworthiness
from conftest import (AUTOENCODER_MANIFOLD_COMBINATIONS,
                      RANDOM_SEED,
                      DATASET_FIXTURES)
from utils import save_experiment, create_subset


@pytest.mark.fast
@pytest.mark.parametrize("dataset_name,dataset_fixture", DATASET_FIXTURES)
@pytest.mark.parametrize("autoencoder_type,manifold_alg", AUTOENCODER_MANIFOLD_COMBINATIONS)
@pytest.mark.parametrize("size_label,train_subset_size,test_subset_size", [
    ("small", 300, 50),
    ("large", 1000, 200),
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
    dataset_fixture,
    request
):
    ((data_train, labels_train), (data_test, labels_test)
     ) = request.getfixturevalue(dataset_fixture)
    data_train_small, labels_train_small = create_subset(
        data_train, train_subset_size, labels_train, seed=RANDOM_SEED)
    data_test_small, labels_test_small = create_subset(
        data_test, test_subset_size, labels_test, seed=RANDOM_SEED)

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
    os.makedirs(results_subpath, exist_ok=True)
    test_name = f"{size_label}_{autoencoder_type}_{manifold_alg}".lower()
    csv_filename = f"tests_{dataset_name}_output.csv"
    embeddings_filename = f"{dataset_name}_{size_label}_{autoencoder_type}_{manifold_alg}_embeddings.pkl".lower(
    )
    labels_filename = f"{dataset_name}_{size_label}_{autoencoder_type}_{manifold_alg}_labels.pkl".lower()

    save_experiment(
        os.path.join(results_subpath, csv_filename),
        os.path.join(results_subpath, embeddings_filename),
        os.path.join(results_subpath, labels_filename),
        test_name,
        output_train,
        labels_train_small,
        output_test,
        labels_test_small,
        trustworthiness_train,
        trustworthiness_test,
        elapsed_fit_train,
        elapsed_transform_train,
        elapsed_transform_test
    )
