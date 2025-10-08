# Imagen base con Python 3.12
FROM python:3.12-slim

# Crear y establecer directorio de trabajo
WORKDIR /gia_ml3_practica1

# Copiar requirements y luego instalar (capa cacheable)
COPY requirements.txt /gia_ml3_practica1/requirements.txt

# Instalar librerías utilizando requirements.txt
RUN apt-get update && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /gia_ml3_practica1/requirements.txt && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /root/.cache/pip

# Copiar el código fuente a la imagen
COPY src/ /gia_ml3_practica1/src/
# Copiar los tests unitarios a la imagen
COPY tests/ /gia_ml3_practica1/tests/

# Punto de entrada por defecto (no ejecuta nada por defecto)
CMD ["/bin/bash"]
