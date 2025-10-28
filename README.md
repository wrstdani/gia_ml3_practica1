# Práctica 1 - Aprendizaje Automático 3
> Repositorio que contiene el código fuente y demás documentos de la práctica 1 de la asignatura Aprendizaje Automático 3.

---
## Estructura del proyecto🏗️
```bash
.
├── artifacts
│   ├── cifar_10_small
│   ├── Fashion-MNIST
│   ├── Glass-Classification
│   └── mnist_784
├── data
│   ├── fashion-mnist_label_dict.json
│   ├── fashion-mnist_test.csv
│   └── fashion-mnist_train.csv
├── Dockerfile
├── entrypoint.sh
├── gia_ml3_practica1_memoria.pdf
├── LICENSE
├── notebooks
│   ├── datasets.ipynb
│   └── graphs.ipynb
│  
├── PracticaObligatoria1.pdf  # Enunciado de la práctica
├── pyproject.toml  # Configuración del proyecto Python
├── pytest.ini
├── README.md
├── requirements-dev.txt  # Dependencias de desarrollador
├── requirements.txt  # Dependencias
├── src
│   ├── autoencoder.py
│   ├── denoising_sparse_autoencoder.py
│   ├── __init__.py
│   ├── linear_autoencoder.py
│   ├── linear_sparse_autoencoder.py
│   ├── mixed_manifold_detector.py
│   ├── utils.py
│   └── variational_autoencoder.py
└── tests
    ├── conftest.py
    ├── test_dimensionality.py  # Tests para comprobar la dimensionalidad de los outputs
    ├── test_metrics_fullsets.py  # Testing con conjuntos de datos completos
    └── test_metrics_subsets.py  # Testing con subconjuntos de datos
```

---
## Ejecución del proyecto
- Es importante que, si se quiere utilizar los conjuntos *train* y *test* de **Fashion MNIST** que se encuentran en data/, utilices `git lfs pull` dentro del repositorio tras clonarlo.
- Para ejecutar este proyecto se recomienda utilizar **Python 3.12** con las librerías (y sus respectivas versiones) especificadas en `./requirements.txt`. Tenemos tres opciones para hacerlo:
    - Venv🐍:
        ```bash
        venv .venv && \
            source .venv/bin/activate && \
            pip install --upgrade pip setuptools wheel && \
            pip install -r requirements.txt && \
            pip install -e . && \
            mixed_manifold_detector </path/to/train.csv> </path/to/test.csv>
        ```

    - Conda🐍:
        ```bash
        conda create -n ml3_practica1 python=3.12 && \
            conda activate ml3_practica1 && \
            pip install -r requirements.txt && \
            pip install -e . && \
            mixed_manifold_detector </path/to/train.csv> </path/to/test.csv>
        ```

    - Docker🐋:
        ```bash
        docker build -t "gia_ml3_practica1:1.4" . && \
            mkdir ./artifacts && \
            docker run --rm \
                -v "$(pwd)/artifacts:/gia_ml3_practica1/artifacts" \
                -v "</path/to/train.csv>:/gia_ml3_practica1/data/train.csv" \
                -v "</path/to/test.csv>:/gia_ml3_practica1/data/test.csv" \
                --user "$(id -u)":"$(id -g)" \
                gia_ml3_practica1:1.4 /gia_ml3_practica1/data/train.csv /gia_ml3_practica1/data/test.csv
        ```
