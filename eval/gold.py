"""Acceso al set dorado de un proyecto y su partición fija.

El set se parte en dos por el nombre de cada imagen: una parte para calibrar la confianza y los
umbrales, y otra para medir. Así nunca se evalúa sobre imágenes con las que se ajustó algo.
La partición es estable: no depende del orden ni de una semilla.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from autolabel.config import ProjectConfig

Part = Literal["all", "calib", "eval"]
_CALIB_SHARE = 4          # de cada 10 imágenes, 4 van a calibración


def part_of(name: str) -> Literal["calib", "eval"]:
    digest = int(hashlib.sha1(name.encode()).hexdigest(), 16)
    return "calib" if digest % 10 < _CALIB_SHARE else "eval"


def load_gold(cfg: ProjectConfig, part: Part = "all") -> dict:
    gold = json.loads((cfg.root / "gold.json").read_text(encoding="utf-8"))
    if part == "all":
        return gold
    return {name: entry for name, entry in gold.items() if part_of(name) == part}


def suffix(part: Part) -> str:
    """Sufijo para los archivos de caché, para que las partes no se pisen."""
    return "" if part == "all" else f"_{part}"
