"""
Testing para dimensionalidad de la salida de los autoencoders y los detectores
"""

import pytest
from conftest import AUTOENCODER_TYPES, MANIFOLD_ALGORITHMS


@pytest.mark.parametrize("autoencoder_type", AUTOENCODER_TYPES)
def test_output_dimensionality_autoencoder(autoencoder_type, autoencoder_factory, sample_data):
    """
    Comprueba que la dimensionalidad de la salida de los distintos tipos de autoencoders
    implementados sea correcta.
    """
    autoencoder = autoencoder_factory(autoencoder_type, sample_data.shape[1])
    autoencoder.fit(sample_data)
    output = autoencoder.transform(sample_data)

    assert output.shape[0] == sample_data.shape[0], \
        f"El número de patrones es {sample_data.shape[0]} y transform devuelve {output.shape[0]} representaciones latentes."
    assert output.shape[1] == autoencoder.latent_dim, \
        f"Según el modelo, la dimensión de las representaciones latentes debería ser {autoencoder.latent_dim} y el output es {output.shape[1]}-dimensional."


@pytest.mark.parametrize("manifold_alg", MANIFOLD_ALGORITHMS)
def test_output_dimensionality_mixed_manifold_detector(manifold_alg, mixed_manifold_detector_factory, sample_data):
    """
    Comprueba que la dimensionalidad de la salida del sistema de detección sea correcta.
    """
    detector = mixed_manifold_detector_factory(
        sample_data.shape[1], manifold_alg)
    output = detector.fit_transform(sample_data)
    assert output.shape[0] == sample_data.shape[0], \
        f"El número de patrones es {sample_data.shape[0]} y transform devuelve {output.shape[0]} representaciones 2D."
    assert output.shape[1] == 2, \
        f"El output final del sistema debe ser 2D y el output es {output.shape[1]}-dimensional."
