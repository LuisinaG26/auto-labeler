# Imagen GPU de autolabel: servidores MCP que usan modelos en CUDA.
# Las ruedas de torch traen su propio runtime CUDA, el host solo aporta el driver.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/cache/huggingface

# Librerías de sistema que necesita OpenCV en modo headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# torch con CUDA 12.4: funciona con drivers que soporten 12.4 o superior
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

COPY docker/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install -r /tmp/requirements-gpu.txt

WORKDIR /app
# El código se monta como volumen en desarrollo (ver docker-compose.yml)
CMD ["python", "-c", "import torch; print('cuda:', torch.cuda.is_available())"]
