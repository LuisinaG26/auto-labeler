"""Servidor MCP del detector.

Es sin estado respecto al proyecto: cada llamada trae las clases, los prompts y
los umbrales, así un mismo servidor atiende a cualquier proyecto. Lo único que
guarda es el modelo cargado, de a uno por vez por el límite de VRAM.

Variables de entorno:
    AUTOLABEL_ALLOWED_ROOTS  carpetas desde las que se pueden leer imágenes, separadas por ':'
    MCP_TRANSPORT            'stdio' (por defecto) o 'streamable-http'
    MCP_HOST, MCP_PORT       para streamable-http
"""
from __future__ import annotations

import gc
import logging
import os
import threading
import time

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from autolabel.detectors import Detector, Query, available_backends, create_detector
from autolabel.imaging import load_image
from services._common import safe_path

log = logging.getLogger("mcp_detector")

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

mcp = FastMCP(
    "autolabel-detector",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8001")),
)


# ------------------------------------------------------------------ modelo en memoria


class _ModelSlot:
    """Mantiene un solo detector cargado. Si piden otro, libera el anterior."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple[str, str] | None = None
        self._detector: Detector | None = None

    def get(self, backend: str, model_id: str) -> Detector:
        with self._lock:
            if self._key != (backend, model_id):
                self._release()
                log.info("Cargando %s / %s", backend, model_id)
                self._detector = create_detector(backend, model_id)
                self._key = (backend, model_id)
            assert self._detector is not None
            return self._detector

    def _release(self) -> None:
        self._detector, self._key = None, None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


_slot = _ModelSlot()


# ------------------------------------------------------------------ esquemas


class QueryIn(BaseModel):
    name: str = Field(description="Nombre de la clase en el proyecto")
    prompt: str = Field(description="Texto que entiende el detector")


class DetectionOut(BaseModel):
    cls: str
    box: list[float] = Field(description="[x1, y1, x2, y2] en píxeles de la imagen original")
    score: float


class DetectResult(BaseModel):
    image_size: list[int] = Field(description="[ancho, alto]")
    detections: list[DetectionOut]
    elapsed_ms: float


# ------------------------------------------------------------------ herramientas


@mcp.tool()
def list_backends() -> list[str]:
    """Backends de detección disponibles."""
    return available_backends()


@mcp.tool()
def detect(
    image_path: str,
    queries: list[QueryIn],
    backend: str = "grounding_dino",
    model_id: str = DEFAULT_MODEL,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_dets: int = 300,
    tiles: int = 1,
) -> DetectResult:
    """Detecta en una imagen los objetos descritos por cada consulta de texto."""
    path = safe_path(image_path)
    detector = _slot.get(backend, model_id)
    start = time.perf_counter()
    img = load_image(path)   # con la orientación EXIF aplicada: las cajas son sobre esta imagen
    dets = detector.detect(
        img,
        [Query(q.name, q.prompt) for q in queries],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        nms_iou=nms_iou,
        max_dets=max_dets,
        tiles=tiles,
    )
    return DetectResult(
        image_size=[img.width, img.height],
        detections=[DetectionOut(cls=d.cls, box=list(d.box), score=d.score) for d in dets],
        elapsed_ms=(time.perf_counter() - start) * 1000,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))  # type: ignore[arg-type]
