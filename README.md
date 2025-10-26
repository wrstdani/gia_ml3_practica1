# Práctica 1 - Aprendizaje Automático 3
> Repositorio que contiene el código fuente y demás documentos de la práctica 1 de la asignatura Aprendizaje Automático 3.

---
## Estructura del proyecto🏗️
```bash
.
├── Dockerfile
├── LICENSE
├── PracticaObligatoria1.pdf  # Enunciado de la práctica
├── README.md
├── data
│   ├── fashion-mnist_label_dict.json  # Diccionario para convertir índices a etiquetas categóricas
│   ├── fashion-mnist_test.csv  # Conjunto de test de Fashion MNIST
│   └── fashion-mnist_train.csv  # Conjunto de entrenamiento de Fashion MNIST
├── entrypoint.sh  # Entrypoint para ejecutar el script principal del proyecto
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
├── src
│   ├── autoencoder.py
│   ├── denoising_sparse_autoencoder.py
│   ├── linear_autoencoder.py
│   ├── linear_sparse_autoencoder.py
│   ├── mixed_manifold_detector.py
│   ├── utils.py
│   └── variational_autoencoder.py
└── tests
    ├── conftest.py  # Configuración de tests
    ├── test_dimensionality.py  # Tests de dimensionalidad de salida
    └── test_metrics_subsets.py  # Tests de métricas con subconjuntos de datos
```

---
## Ejecución del proyecto
- Para ejecutar este proyecto se recomienda utilizar **Python 3.12** con las librerías (y sus respectivas versiones) especificadas en `./requirements.txt`. Tenemos dos opciones para hacerlo:
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
