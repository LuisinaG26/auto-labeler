"""Modelos de visión-lenguaje. El proveedor se elige en ``project.yaml``."""
from __future__ import annotations

from autolabel.config import VlmConfig
from autolabel.vlm.base import VisionLanguageModel, VlmError

__all__ = ["VisionLanguageModel", "VlmError", "create_vlm"]


def create_vlm(cfg: VlmConfig) -> VisionLanguageModel:
    if cfg.provider == "ollama":
        from autolabel.vlm.ollama import OllamaVlm

        return OllamaVlm(cfg.model, cfg.base_url, max_image_side=cfg.max_image_side)
    raise ValueError(f"Proveedor de VLM desconocido: {cfg.provider!r}")
