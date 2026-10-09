"""Servidor MCP del filtro de calidad de imágenes.

Sin estado y sin modelos: cada llamada trae los cortes a aplicar, así un mismo
servidor atiende a cualquier proyecto. Los duplicados se detectan dentro del lote
recibido, por lo que conviene enviar todas las imágenes de una vez.

Variables de entorno:
    AUTOLABEL_ALLOWED_ROOTS  carpetas desde las que se pueden leer imágenes, separadas por ':'
    MCP_TRANSPORT            'stdio' (por defecto) o 'streamable-http'
    MCP_HOST, MCP_PORT       para streamable-http
"""
from __future__ import annotations

import logging
import os
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from autolabel.config import QualityConfig
from autolabel.quality import Assessment, assess_batch
from services._common import safe_path

mcp = FastMCP(
    "autolabel-quality",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8002")),
)


class MetricsOut(BaseModel):
    width: int
    height: int
    blur: float = Field(description="Nitidez: más alto, más nítida")
    brightness: float
    clipped: float
    phash: str = Field(description="Hash perceptual de 64 bits en hexadecimal")


class AssessmentOut(BaseModel):
    image: str
    verdict: Literal["keep", "drop", "unsure"]
    reasons: list[str]
    metrics: MetricsOut | None


def _to_out(a: Assessment) -> AssessmentOut:
    metrics = None
    if a.metrics:
        m = a.metrics
        metrics = MetricsOut(
            width=m.width, height=m.height, blur=m.blur, brightness=m.brightness,
            clipped=m.clipped, phash=f"{m.phash:016x}",
        )
    return AssessmentOut(image=a.image, verdict=a.verdict, reasons=list(a.reasons), metrics=metrics)


@mcp.tool()
def assess_images(image_paths: list[str], params: QualityConfig | None = None) -> list[AssessmentOut]:
    """Evalúa la calidad de un lote de imágenes y devuelve `keep`, `drop` o `unsure` con el motivo.

    Aplica resolución mínima, exposición, desenfoque y duplicados dentro del lote.
    Si no se envían `params`, se usan los cortes por defecto.
    """
    paths = [safe_path(p) for p in image_paths]
    return [_to_out(a) for a in assess_batch(paths, params or QualityConfig())]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    mcp.run(transport=os.environ.get("MCP_TRANSPORT", "stdio"))  # type: ignore[arg-type]
