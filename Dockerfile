# Imagen base con Python 3.12
FROM python:3.12-slim

# Crear y establecer directorio de trabajo
WORKDIR /gia_ml3_practica1

RUN mkdir -p /gia_ml3_practica1/artifacts && \
    chown root:root /gia_ml3_practica1/artifacts -R && \
    chmod -R a+rwX /gia_ml3_practica1/artifacts
VOLUME ["/gia_ml3_practica1/artifacts"]

# Copiar el proyecto a la imagen
COPY . /gia_ml3_practica1

# Instalar librerías utilizando requirements.txt
RUN apt-get update && \
    python -m venv /gia_ml3_practica1/venv && \
    /gia_ml3_practica1/venv/bin/pip install --upgrade pip setuptools wheel && \
    /gia_ml3_practica1/venv/bin/pip install --no-cache-dir -r /gia_ml3_practica1/requirements.txt && \
    /gia_ml3_practica1/venv/bin/pip install --no-cache-dir -r /gia_ml3_practica1/requirements-dev.txt && \
    /gia_ml3_practica1/venv/bin/pip install -e "/gia_ml3_practica1[dev]"

# Punto de entrada por defecto
RUN chmod +x /gia_ml3_practica1/entrypoint.sh
ENTRYPOINT ["/gia_ml3_practica1/entrypoint.sh"]
CMD []
