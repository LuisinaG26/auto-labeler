"""Lector de datasets en formato DatasetNinja (Supervisely).

Estructura: ``<root>/<split>/img/<archivo>.jpg`` y ``<root>/<split>/ann/<archivo>.jpg.json``.
Admite polígonos y rectángulos; ambos se reducen a cajas.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from autolabel.geometry import polygon_to_box

_SUPPORTED_GEOMETRIES = frozenset({"polygon", "rectangle"})


@dataclass(frozen=True, slots=True)
class LabeledBox:
    label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class NinjaItem:
    image_path: Path
    width: int
    height: int
    objects: tuple[LabeledBox, ...]


def iter_split(root: str | Path, split: str) -> Iterator[NinjaItem]:
    """Recorre un split en orden alfabético, de forma perezosa."""
    split_dir = Path(root) / split
    for ann_path in sorted((split_dir / "ann").glob("*.json")):
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        objects = tuple(
            LabeledBox(o["classTitle"], tuple(polygon_to_box(o["points"]["exterior"])))
            for o in data.get("objects", [])
            if o.get("geometryType") in _SUPPORTED_GEOMETRIES and len(o["points"]["exterior"]) >= 2
        )
        # "IMG_0983.jpg.json" -> stem "IMG_0983.jpg"
        yield NinjaItem(
            image_path=split_dir / "img" / ann_path.stem,
            width=data["size"]["width"],
            height=data["size"]["height"],
            objects=objects,
        )
