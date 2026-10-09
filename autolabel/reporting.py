"""Salidas para mirar con los ojos: hojas de contacto con la medida de cada imagen."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from autolabel.imaging import load_image

Color = tuple[int, int, int]
GREEN: Color = (0, 255, 0)


def detections_sheet(
    items: Sequence[tuple[str | Path, int, int, Sequence[tuple[Sequence[float], str, Color]]]],
    out_path: str | Path,
    *,
    columns: int = 4,
    thumb: int = 320,
) -> None:
    """Hoja de miniaturas con las cajas dibujadas, para ver el resultado sin abrir otra herramienta.

    Cada elemento es ``(ruta, ancho, alto, [(caja, texto, color), ...])`` con las coordenadas
    sobre la imagen ya girada, que es como las devuelve el detector.
    """
    rows = -(-len(items) // columns)
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (30, 30, 30))
    for k, (path, width, height, boxes) in enumerate(items):
        img = load_image(path, max_side=thumb)
        sx, sy = img.width / width, img.height / height
        draw = ImageDraw.Draw(img)
        for box, text, color in boxes:
            x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            draw.text((x1 + 2, y1 + 1), text, fill=(255, 255, 0))
        cell = Image.new("RGB", (thumb, thumb), (30, 30, 30))
        cell.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
        sheet.paste(cell, ((k % columns) * thumb, (k // columns) * thumb))
    sheet.save(out_path, quality=90)


def contact_sheet(
    paths: Sequence[str | Path],
    labels: Sequence[str],
    out_path: str | Path,
    *,
    columns: int = 5,
    thumb: int = 240,
) -> None:
    """Guarda una grilla de miniaturas, cada una con su etiqueta escrita encima."""
    if len(paths) != len(labels):
        raise ValueError("paths y labels deben tener la misma longitud")
    rows = -(-len(paths) // columns)
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (30, 30, 30))
    for k, (path, label) in enumerate(zip(paths, labels, strict=True)):
        img = load_image(path, max_side=thumb)
        cell = Image.new("RGB", (thumb, thumb), (30, 30, 30))
        cell.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
        draw = ImageDraw.Draw(cell)
        draw.rectangle((0, 0, thumb, 14), fill=(0, 0, 0))
        draw.text((3, 1), label, fill=(255, 255, 0))
        sheet.paste(cell, ((k % columns) * thumb, (k // columns) * thumb))
    sheet.save(out_path, quality=90)
