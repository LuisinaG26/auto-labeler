# Imagen liviana de autolabel: orquestador, calidad de imagen y dataset. Sin GPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-light.txt /tmp/requirements-light.txt
RUN pip install -r /tmp/requirements-light.txt

WORKDIR /app
CMD ["python", "-c", "import langgraph, mcp; print('ok')"]
