"""Contrato común de los backends de detección guiada por texto."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True, slots=True)
class Query:
    """Una clase a buscar: nombre del proyecto y texto que entiende el detector."""

    name: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Detection:
    cls: str
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 en píxeles de la imagen original
    score: float


@runtime_checkable
class Detector(Protocol):
    """Todo backend recibe imagen y consultas, y devuelve cajas con clase y score."""

    def detect(
        self,
        image: Image.Image,
        queries: Sequence[Query],
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        max_dets: int = 300,
        tiles: int = 1,
    ) -> list[Detection]: ...
