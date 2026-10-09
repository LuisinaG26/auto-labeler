"""Registro de backends de detección.

Los backends se importan de forma perezosa para no cargar torch ni transformers
cuando solo se necesita el contrato.
"""
from __future__ import annotations

from importlib import import_module

from autolabel.detectors.base import Detection, Detector, Query

__all__ = ["Detection", "Detector", "Query", "available_backends", "create_detector"]

# nombre público -> (módulo, clase)
_BACKENDS: dict[str, tuple[str, str]] = {
    "grounding_dino": ("autolabel.detectors.grounding_dino", "GroundingDinoDetector"),
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def create_detector(backend: str, model_id: str) -> Detector:
    try:
        module_name, class_name = _BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"Backend desconocido {backend!r}. Disponibles: {available_backends()}"
        ) from None
    return getattr(import_module(module_name), class_name)(model_id)
