"""Carga de imágenes con la orientación EXIF aplicada.

Las fotos de celular suelen guardarse en horizontal con una marca EXIF que indica
cómo rotarlas. Los datasets y las herramientas de etiquetado trabajan sobre la
imagen ya rotada, así que todo el flujo debe usar este cargador y no ``Image.open``.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})


def list_images(folder: Path) -> list[Path]:
    """Imágenes de una carpeta, en orden alfabético."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


_EXIF_ORIENTATION = 274
_SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})   # rotaciones de 90 grados


def image_size(path: str | Path) -> tuple[int, int]:
    """Ancho y alto con la orientación aplicada, leyendo solo el encabezado."""
    with Image.open(path) as img:
        width, height = img.size
        if img.getexif().get(_EXIF_ORIENTATION, 1) in _SWAPPED_ORIENTATIONS:
            width, height = height, width
    return width, height


def load_image(path: str | Path, max_side: int | None = None) -> Image.Image:
    """Abre la imagen en RGB, con la orientación aplicada y, opcional, con lado mayor <= max_side.

    Para JPEG pide la decodificación directa a escala reducida (``draft``), que es
    mucho más rápida que decodificar a tamaño completo y luego reducir.
    """
    img = Image.open(path)
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img.draft("RGB", (round(img.width * scale), round(img.height * scale)))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    return img
