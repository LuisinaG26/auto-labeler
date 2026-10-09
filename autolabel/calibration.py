"""Confianza combinada de una caja, calibrada contra un set dorado.

Combina las señales de la verificación en una probabilidad de que la caja sea correcta,
con una regresión logística pequeña. Se ajusta una vez por proyecto con su set dorado y
se guarda en ``out/calibration.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from autolabel.verification import BoxVerdict

FEATURES = ("det_score", "p_one", "p_several", "group_geometry")
CLASS_FEATURES = ("p_class", "class_margin")


def features_of(v: BoxVerdict) -> list[float]:
    """Variables con las que se estima si la caja es un objeto correcto."""
    return [v.det_score, v.p_one, v.p_several, float(v.group_geometry)]


def class_features_of(v: BoxVerdict) -> list[float]:
    """Variables con las que se estima si la clase elegida por el VLM es correcta."""
    probs = sorted(v.class_probs.values(), reverse=True)
    if not probs:
        return [0.0, 0.0]
    second = probs[1] if len(probs) > 1 else 0.0
    return [probs[0], probs[0] - second]


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass(frozen=True, slots=True)
class ConfidenceModel:
    """Regresión logística ya ajustada, con las variables estandarizadas."""

    features: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = (np.atleast_2d(x) - np.array(self.mean)) / np.array(self.scale)
        return _sigmoid(z @ np.array(self.weights) + self.bias)

    def to_dict(self) -> dict:
        return {"features": self.features, "mean": self.mean, "scale": self.scale,
                "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_dict(cls, d: dict) -> ConfidenceModel:
        return cls(tuple(d["features"]), tuple(d["mean"]), tuple(d["scale"]), tuple(d["weights"]), d["bias"])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ConfidenceModel:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class Calibration:
    """Confianza de una caja verificada: probabilidad de que sea un objeto y de que su clase sea correcta.

    ``cls`` es opcional. Sin él, o en un proyecto de una sola clase, la clase no resta confianza.
    """

    box: ConfidenceModel
    cls: ConfidenceModel | None = None

    def box_confidence(self, v: BoxVerdict) -> float:
        return float(self.box.predict(np.array(features_of(v)))[0])

    def class_confidence(self, v: BoxVerdict) -> float:
        if self.cls is None:
            return 1.0
        if not v.class_probs:               # la clase no se evaluó: no es un objeto reconocible
            return 0.0
        return float(self.cls.predict(np.array(class_features_of(v)))[0])

    def confidence(self, v: BoxVerdict) -> float:
        return self.box_confidence(v) * self.class_confidence(v)

    def signature(self) -> list:
        """Identifica la calibración, para saber si un resultado guardado sigue siendo válido."""
        cls_part = None if self.cls is None else [self.cls.weights, self.cls.bias]
        return [self.box.weights, self.box.bias, cls_part]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"box": self.box.to_dict(), "class": None if self.cls is None else self.cls.to_dict()}, indent=1
        ), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        d = json.loads(path.read_text(encoding="utf-8"))
        if "box" not in d:                  # formato anterior: solo el modelo de la caja
            return cls(ConfidenceModel.from_dict(d))
        return cls(ConfidenceModel.from_dict(d["box"]),
                   ConfidenceModel.from_dict(d["class"]) if d.get("class") else None)


def fit_logistic(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    feature_names: tuple[str, ...] = FEATURES,
    l2: float = 1.0,
    iterations: int = 50,
) -> ConfidenceModel:
    """Regresión logística con regularización L2, por Newton (mínimos cuadrados reponderados).

    Las variables se estandarizan para que la penalización las trate por igual.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-9, x.std(axis=0), 1.0)
    z = (x - mean) / scale
    design = np.hstack([z, np.ones((len(z), 1))])
    penalty = l2 * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0                         # el sesgo no se penaliza
    w = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = _sigmoid(design @ w)
        gradient = design.T @ (p - y) + penalty @ w
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    if len(feature_names) != x.shape[1]:
        raise ValueError("feature_names no coincide con las columnas de x")
    return ConfidenceModel(
        features=tuple(feature_names), mean=tuple(mean.tolist()), scale=tuple(scale.tolist()),
        weights=tuple(w[:-1].tolist()), bias=float(w[-1]),
    )


def roc_auc(scores: NDArray[np.float64], labels: NDArray[np.bool_]) -> float:
    """Probabilidad de que un positivo puntúe más alto que un negativo (con empates a la mitad)."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rango promedio de cada valor: los empates reparten el rango a partes iguales
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    average_rank = (last - counts + 1 + last) / 2
    ranks = average_rank[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
