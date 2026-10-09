"""Contrato de los modelos de visión-lenguaje (VLM), independiente del proveedor.

Hay dos operaciones, las que necesita el flujo:
  * ``choose``: elegir una opción y devolver la probabilidad de cada una, para poder usarla
    como confianza.
  * ``ask_json``: respuesta estructurada, para cuando hace falta además un texto (un motivo).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from PIL import Image


class VlmError(RuntimeError):
    """El modelo respondió algo que no se pudo interpretar."""


@runtime_checkable
class VisionLanguageModel(Protocol):
    def choose(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        options: Sequence[str],
        *,
        system: str | None = None,
    ) -> dict[str, float]:
        """Probabilidad de cada opción, que suma 1. Si el proveedor no entrega probabilidades,
        devuelve 1.0 para la opción elegida y 0.0 para el resto."""
        ...

    def ask_json(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
    ) -> dict:
        """Respuesta que cumple el esquema JSON dado."""
        ...
