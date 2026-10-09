"""Utilidades compartidas por los servidores MCP."""
from __future__ import annotations

import os
from pathlib import Path


def allowed_roots() -> tuple[Path, ...]:
    """Carpetas desde las que se leen imágenes (AUTOLABEL_ALLOWED_ROOTS, separadas por ':')."""
    raw = os.environ.get("AUTOLABEL_ALLOWED_ROOTS", "/app/projects:/data")
    return tuple(Path(p).resolve() for p in raw.split(":") if p)


def safe_path(path: str, roots: tuple[Path, ...] | None = None) -> Path:
    """Resuelve una ruta y exige que sea un archivo dentro de las carpetas permitidas."""
    roots = roots if roots is not None else allowed_roots()
    p = Path(path).resolve()
    if not any(p.is_relative_to(root) for root in roots):
        raise ValueError(f"Ruta fuera de las carpetas permitidas: {path}")
    if not p.is_file():
        raise FileNotFoundError(path)
    return p
