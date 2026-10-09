"""Arma un set dorado reducido a partir de un dataset DatasetNinja.

Toma una muestra con semilla fija, reduce las imágenes y convierte las etiquetas
de origen a las clases del proyecto. No modifica el dataset original.

Uso:
    python -m eval.make_gold --project projects/tomates-gold \\
        --source /data/dataset-tomates/laborotomato-DatasetNinja --split Test --n 100
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter

from PIL import Image

from autolabel.config import load_project
from autolabel.datasets.ninja import NinjaItem, iter_split
from autolabel.geometry import clip_boxes, scale_boxes
from autolabel.imaging import load_image

log = logging.getLogger("make_gold")


def _load_scaled(item: NinjaItem, max_side: int) -> tuple[Image.Image, float, float]:
    """Imagen reducida y factores de escala hacia las coordenadas de las etiquetas.

    Las etiquetas del dataset se usan tal cual vienen, sin modificarlas. Lo que se
    adapta es la imagen: se gira según EXIF para quedar en el mismo marco en que se
    dibujaron. Si aun así las proporciones no coinciden, falla en vez de producir
    cajas desalineadas.
    """
    img = load_image(item.image_path, max_side)
    sx, sy = img.width / item.width, img.height / item.height
    if abs(sx - sy) > 0.01:
        raise ValueError(
            f"{item.image_path.name}: la imagen {img.size} no coincide en proporción con "
            f"las etiquetas ({item.width}x{item.height})"
        )
    return img, sx, sy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--source", required=True, help="Carpeta raíz del dataset DatasetNinja")
    ap.add_argument("--split", nargs="+", default=["Test"], help="Uno o más splits del dataset origen")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=500, help="Lado mayor tras reducir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_project(args.project)
    label_map = cfg.label_map()
    items = [item for split in args.split for item in iter_split(args.source, split)]
    sample = random.Random(args.seed).sample(items, min(args.n, len(items)))
    sample.sort(key=lambda it: it.image_path.name)

    cfg.images_dir.mkdir(parents=True, exist_ok=True)
    gold: dict[str, dict] = {}
    per_class: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()

    for item in sample:
        img, sx, sy = _load_scaled(item, args.max_side)
        name = item.image_path.with_suffix(".jpg").name
        img.save(cfg.images_dir / name, quality=92)

        kept = [(o, label_map.get(o.label)) for o in item.objects]
        unmapped.update(o.label for o, cls in kept if cls is None)
        mapped = [(o, cls) for o, cls in kept if cls is not None]
        boxes = clip_boxes(scale_boxes([o.box for o, _ in mapped], sx, sy), *img.size)
        gold[name] = {
            "size": list(img.size),
            "boxes": [
                {"cls": cls, "box": box.round(1).tolist()}
                for (_, cls), box in zip(mapped, boxes, strict=True)
            ],
        }
        per_class.update(cls for _, cls in mapped)

    (cfg.root / "gold.json").write_text(json.dumps(gold, indent=1), encoding="utf-8")
    log.info("Imágenes: %d de %d (semilla %d)", len(sample), len(items), args.seed)
    log.info("Cajas por clase: %s", dict(per_class))
    if unmapped:
        log.warning("Etiquetas de origen sin clase en el proyecto: %s", dict(unmapped))


if __name__ == "__main__":
    main()
