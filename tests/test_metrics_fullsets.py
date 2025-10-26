import os
import pytest
import time
from sklearn.manifold import trustworthiness
from tests.conftest import (AUTOENCODER_MANIFOLD_COMBINATIONS,
                            DATASET_FIXTURES)
from src.utils import save_experiment


@pytest.mark.parametrize("dataset_name,dataset_fixture", DATASET_FIXTURES.items())
@pytest.mark.parametrize("autoencoder_type,manifold_alg", AUTOENCODER_MANIFOLD_COMBINATIONS)
def test_metrics_fullsets(
    dataset_name,
    autoencoder_factory,
    mixed_manifold_detector_factory,
    autoencoder_type,
    manifold_alg,
    results_path,
    dataset_fixture,
    request
):
    data_train, data_test = request.getfixturevalue(dataset_fixture)

    autoencoder = autoencoder_factory(
        autoencoder_type, data_train.shape[1])
    detector = mixed_manifold_detector_factory(
        data_test.shape[1], manifold_alg, autoencoder=autoencoder)

    start_fit_train = time.time()
    detector.fit(data_train)
    elapsed_fit_train = time.time() - start_fit_train

    start_transform_train = time.time()
    output_train = detector.transform(data_train)
    elapsed_transform_train = time.time() - start_transform_train
    trustworthiness_train = trustworthiness(data_train, output_train)

    start_transform_test = time.time()
    output_test = detector.transform(data_test)
    elapsed_transform_test = time.time() - start_transform_test
    trustworthiness_test = trustworthiness(data_test, output_test)

    results_subpath = os.path.join(results_path, f"{dataset_name}")
    os.makedirs(results_subpath, exist_ok=True)
    test_name = f"{dataset_name}_fullset_{autoencoder_type}_{manifold_alg}".lower()
    csv_filename = f"tests_{dataset_name}_output.csv"
    embeddings_filename = f"{dataset_name}_fullset_{autoencoder_type}_{manifold_alg}.pkl".lower()

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
