"""Métricas de detección: AP a un IoU fijo, precisión, recall y F1.

El emparejamiento es voraz por score, como en COCO. Las cajas de referencia
marcadas como ignoradas (por ejemplo, demasiado chicas) no cuentan como
pendientes y las detecciones que caen sobre ellas no cuentan como falsos positivos.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from autolabel.geometry import as_boxes, iou_matrix


@dataclass(frozen=True, slots=True)
class Curve:
    """Detecciones de todas las imágenes, ordenadas por score descendente."""

    scores: NDArray[np.float64]
    tp: NDArray[np.bool_]
    n_gt: int

    def _cumulative(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        tp = np.cumsum(self.tp, dtype=np.float64)
        fp = np.cumsum(~self.tp, dtype=np.float64)
        return tp, fp

    def ap(self) -> float:
        """AP con interpolación de todos los puntos."""
        if self.n_gt == 0 or len(self.scores) == 0:
            return 0.0
        tp, fp = self._cumulative()
        recall = tp / self.n_gt
        precision = tp / (tp + fp)
        # Envolvente monótona de la precisión
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        recall = np.concatenate(([0.0], recall))
        return float(np.sum(np.diff(recall) * precision))

    def at(self, threshold: float) -> dict[str, float]:
        """Precisión, recall y F1 conservando las detecciones con score >= threshold."""
        k = int(np.searchsorted(-self.scores, -threshold, side="right"))
        tp = float(np.sum(self.tp[:k]))
        fp = float(k - tp)
        fn = float(self.n_gt - tp)
        return _prf(tp, fp, fn) | {"threshold": float(threshold)}

    def best_f1(self) -> dict[str, float]:
        """Umbral de score que maximiza el F1."""
        if len(self.scores) == 0:
            return _prf(0.0, 0.0, float(self.n_gt)) | {"threshold": 1.0}
        tp, fp = self._cumulative()
        fn = self.n_gt - tp
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        i = int(np.argmax(f1))
        return _prf(float(tp[i]), float(fp[i]), float(fn[i])) | {"threshold": float(self.scores[i])}


def _prf(tp: float, fp: float, fn: float) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def match_image(
    det_boxes: ArrayLike,
    det_scores: ArrayLike,
    gt_boxes: ArrayLike,
    ignore_boxes: ArrayLike,
    iou_thr: float,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Empareja las detecciones de una imagen. Devuelve scores y si cada una acierta.

    Las detecciones que se solapan con una caja ignorada se descartan.
    """
    det_boxes, gt_boxes, ignore_boxes = as_boxes(det_boxes), as_boxes(gt_boxes), as_boxes(ignore_boxes)
    scores = np.asarray(det_scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    det_boxes, scores = det_boxes[order], scores[order]

    ious = iou_matrix(det_boxes, gt_boxes)
    ignore_hit = (iou_matrix(det_boxes, ignore_boxes) >= iou_thr).any(axis=1)

    taken = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(scores), dtype=bool)
    keep = np.ones(len(scores), dtype=bool)
    for i in range(len(scores)):
        if len(gt_boxes):
            candidates = np.where(taken, -1.0, ious[i])
            j = int(np.argmax(candidates))
            if candidates[j] >= iou_thr:
                taken[j] = True
                tp[i] = True
                continue
        if ignore_hit[i]:
            keep[i] = False
    return scores[keep], tp[keep]


def build_curve(
    images: Iterable[tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]],
    iou_thr: float = 0.5,
) -> Curve:
    """Cada elemento: ``(det_boxes, det_scores, gt_boxes, ignore_boxes)`` de una imagen."""
    all_scores: list[NDArray[np.float64]] = []
    all_tp: list[NDArray[np.bool_]] = []
    n_gt = 0
    for det_boxes, det_scores, gt_boxes, ignore_boxes in images:
        n_gt += len(as_boxes(gt_boxes))
        s, t = match_image(det_boxes, det_scores, gt_boxes, ignore_boxes, iou_thr)
        all_scores.append(s)
        all_tp.append(t)
    scores = np.concatenate(all_scores) if all_scores else np.empty(0)
    tp = np.concatenate(all_tp) if all_tp else np.empty(0, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    return Curve(scores[order], tp[order], n_gt)
