"""Imágenes de referencia por clase.

Son datos del proyecto: ``refs/<clase>/*.png`` dentro de la carpeta del proyecto, o las rutas
listadas en ``classes[].references`` del YAML. Son opcionales: sin ellas el flujo usa solo texto.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from autolabel.config import ClassConfig, ProjectConfig
from autolabel.imaging import load_image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_NEGATIVE_PREFIX = "not_"          # los negativos se nombran not_*.png


def _paths_for(cfg: ProjectConfig, cls: ClassConfig, *, negative: bool) -> list[Path]:
    listed = cls.references.negative if negative else cls.references.positive
    if listed:
        return [cfg.root / p if not Path(p).is_absolute() else Path(p) for p in listed]
    folder = cfg.root / "refs" / cls.name
    if not folder.is_dir():
        return []
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    return [p for p in files if p.name.startswith(_NEGATIVE_PREFIX) == negative]


def load_references(cfg: ProjectConfig, max_per_class: int = 3) -> dict[str, list[Image.Image]]:
    """Referencias positivas por clase. Las clases sin referencias no aparecen."""
    out: dict[str, list[Image.Image]] = {}
    for cls in cfg.classes:
        paths = _paths_for(cfg, cls, negative=False)[:max_per_class]
        if paths:
            out[cls.name] = [load_image(p) for p in paths]
    return out


def reference_sheet(
    refs: dict[str, list[Image.Image]], class_order: list[str], *, cell: int = 160
) -> Image.Image | None:
    """Una sola imagen con una fila por clase y su nombre escrito, para pasarla al VLM.

    Usar una imagen en lugar de una por referencia ahorra memoria de video y contexto, que en
    una GPU de 8 GB es lo escaso, y deja claro qué clase muestra cada fila.
    """
    rows = [(name, refs[name]) for name in class_order if name in refs]
    if not rows:
        return None
    width = max(len(imgs) for _, imgs in rows) * cell
    header = 18
    sheet = Image.new("RGB", (width, len(rows) * (cell + header)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (name, imgs) in enumerate(rows):
        y = r * (cell + header)
        draw.text((4, y + 3), f"class: {name}", fill=(0, 0, 0))
        for c, img in enumerate(imgs):
            thumb = img.copy()
            thumb.thumbnail((cell, cell))
            left = c * cell + (cell - thumb.width) // 2
            top = y + header + (cell - thumb.height) // 2
            sheet.paste(thumb, (left, top))
    return sheet
