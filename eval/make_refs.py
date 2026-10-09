"""Arma imágenes de referencia por clase a partir de un dataset anotado en formato DatasetNinja.

Toma recortes de un split distinto del set dorado, para que ninguna referencia salga de una
imagen de evaluación. Guarda ``refs/<clase>/ok_<n>.png`` dentro de la carpeta del proyecto.

Uso:
    python -m eval.make_refs --project projects/tomates-gold \\
        --source /data/dataset-tomates/laborotomato-DatasetNinja --split Train --per-class 3
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict

from autolabel.config import load_project
from autolabel.datasets.ninja import iter_split
from autolabel.geometry import scale_boxes
from autolabel.imaging import load_image
from autolabel.verification import crop_candidate

log = logging.getLogger("make_refs")

_LOAD_SIDE = 1200          # lado mayor al que se decodifica la imagen origen
_MIN_BOX = 60              # lado menor mínimo de la caja, ya a esa escala, para que el recorte sea nítido
_MAX_FRACTION = 0.6        # una caja que ocupa casi toda la imagen suele ser un grupo o un primer plano raro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--split", default="Train")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_project(args.project)
    label_map = cfg.label_map()
    rng = random.Random(args.seed)

    # Candidatos: (imagen, caja en coordenadas de las etiquetas, clase), sin cargar imágenes todavía
    candidates: dict[str, list] = defaultdict(list)
    for item in iter_split(args.source, args.split):
        longest = max(item.width, item.height)
        for obj in item.objects:
            cls = label_map.get(obj.label)
            if cls is None:
                continue
            w, h = obj.box[2] - obj.box[0], obj.box[3] - obj.box[1]
            scale = _LOAD_SIDE / longest
            if min(w, h) * scale >= _MIN_BOX and max(w, h) <= _MAX_FRACTION * longest:
                candidates[cls].append((item, obj.box))

    for cls in cfg.class_names:
        pool = candidates.get(cls, [])
        if len(pool) < args.per_class:
            raise SystemExit(f"Clase {cls!r}: solo {len(pool)} candidatos, se piden {args.per_class}")
        out_dir = cfg.root / "refs" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("ok_*.png"):
            old.unlink()
        used: set[str] = set()
        picked = 0
        for item, box in rng.sample(pool, len(pool)):
            if item.image_path.name in used:         # una referencia por imagen, para más variedad
                continue
            img = load_image(item.image_path, max_side=_LOAD_SIDE)
            sx, sy = img.width / item.width, img.height / item.height
            scaled = scale_boxes([box], sx, sy)[0]
            crop_candidate(img, scaled, min_side=160, max_side=320).save(out_dir / f"ok_{picked + 1}.png")
            used.add(item.image_path.name)
            picked += 1
            if picked == args.per_class:
                break
        log.info("%s: %d referencias en %s", cls, picked, out_dir)


if __name__ == "__main__":
    main()
