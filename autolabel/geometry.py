"""Utilidades de cajas vectorizadas con NumPy.

Formato único: arrays ``(N, 4)`` con ``[x1, y1, x2, y2]`` en píxeles.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_boxes(boxes: ArrayLike) -> FloatArray:
    """Convierte a un array ``(N, 4)`` de float64, también si la lista está vacía."""
    arr = np.asarray(boxes, dtype=np.float64)
    return arr.reshape(-1, 4)


def polygon_to_box(points: Iterable[Sequence[float]]) -> list[float]:
    pts = np.asarray(list(points), dtype=np.float64)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def box_sizes(boxes: ArrayLike) -> FloatArray:
    """Ancho y alto de cada caja, forma ``(N, 2)``."""
    b = as_boxes(boxes)
    return np.stack([b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def iou_matrix(a: ArrayLike, b: ArrayLike) -> FloatArray:
    """IoU de todas las cajas de ``a`` contra todas las de ``b``, forma ``(len(a), len(b))``."""
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter_wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def clip_boxes(boxes: ArrayLike, width: float, height: float) -> FloatArray:
    b = as_boxes(boxes).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    return b


def scale_boxes(boxes: ArrayLike, sx: float, sy: float) -> FloatArray:
    return as_boxes(boxes) * np.array([sx, sy, sx, sy])
