# Práctica 1 - Aprendizaje Automático 3
> Repositorio que contiene el código fuente y demás documentos de la práctica 1 de la asignatura Aprendizaje Automático 3.

---
## Estructura del proyecto🏗️
```bash
.  
├── data/  # Conjuntos de datos
│   └── ...
├── Dockerfile  # Dockerfile para construir imagen
├── .gitignore
├── LICENSE
├── notebooks/  # Jupyter notebooks para experimentos
│   └── ...
├── README.md
├── requirements.txt  # Librerías de Python utilizadas
├── resources/  # PDF
│   └── PracticaObligatoria1.pdf
├── src/  # Código fuente
│   ├── autoencoder.py
│   └── mixed_manifold_detector.py
└── tests/  # Tests unitarios
    └── ...
```

---
## Ejecución del proyecto
- Para ejecutar este proyecto se recomienda utilizar **Python 3.12** con las librerías (y sus respectivas versiones) especificadas en `./requirements.txt`. Tenemos dos opciones para hacerlo:
    - Conda🐍:
        ```bash
        conda create -n ml3_practica1 python=3.12 && \
            conda activate ml3_practica1 && \
            pip install -r requirements.txt
        ```

    - Docker🐋:
        ```bash
        docker build -t gia_ml3_practica1:0.1 . && \
            docker run --name practica1 --rm -it gia_ml3_practica1:0.1
        ```
        - Por el momento, no hay ningún código funcional, únicamente la estructura del código, por lo que lo único que podemos hacer es acceder a `bash` (como hacemos en el comando de arriba) y visualizar el workspace.
