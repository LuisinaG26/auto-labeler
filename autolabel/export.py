"""Exportación de etiquetas a YOLO.

Las coordenadas son relativas a la imagen con la orientación EXIF aplicada, la misma
que ven los visores y las herramientas de etiquetado.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from autolabel.geometry import clip_boxes
from autolabel.labeling import ImageLabels


def yolo_lines(item: ImageLabels, class_index: dict[str, int]) -> list[str]:
    """Una línea por caja: ``clase cx cy ancho alto`` normalizados a [0, 1]."""
    lines = []
    for d in item.detections:
        x1, y1, x2, y2 = clip_boxes([d.box], item.width, item.height)[0]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        lines.append(
            f"{class_index[d.cls]} {(x1 + x2) / 2 / item.width:.6f} {(y1 + y2) / 2 / item.height:.6f} "
            f"{w / item.width:.6f} {h / item.height:.6f}"
        )
    return lines


def write_yolo(
    results: Sequence[ImageLabels], class_names: Sequence[str], images_dir: Path, out_dir: Path
) -> Path:
    """Escribe ``labels/*.txt``, ``classes.txt``, ``images.txt`` y ``data.yaml``.

    Una imagen sin detecciones genera un archivo vacío, que en YOLO significa "sin objetos".
    """
    class_index = {name: i for i, name in enumerate(class_names)}
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for item in results:
        (labels_dir / f"{Path(item.image).stem}.txt").write_text(
            "".join(line + "\n" for line in yolo_lines(item, class_index)), encoding="utf-8"
        )
    (out_dir / "classes.txt").write_text("".join(f"{n}\n" for n in class_names), encoding="utf-8")
    (out_dir / "images.txt").write_text(
        "".join(f"{images_dir / r.image}\n" for r in results), encoding="utf-8"
    )
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images.txt\nval: images.txt\nnames:\n{names}\n", encoding="utf-8"
    )
    return out_dir
