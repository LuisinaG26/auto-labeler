"""Valida el filtro de calidad con desenfoque artificial sobre el set dorado.

Aplica desenfoque gaussiano creciente a las imágenes nítidas del set dorado y mide
para cada nivel la nitidez del filtro y el rendimiento del detector. Muestra dos cosas:
que la medida responde al desenfoque y que el detector pierde recall a medida que crece,
que es la razón para apartar esas imágenes.

Los puntajes de nitidez dependen de la cámara. Las imágenes del set dorado son mucho más
nítidas que las de otros proyectos, así que esta prueba valida la medida y el efecto en
el detector, no los cortes absolutos de un proyecto.

Uso:
    python -m eval.eval_quality --project projects/tomates-gold
"""
from __future__ import annotations

import argparse
import json
import logging

import cv2
import numpy as np
from PIL import Image

from autolabel.config import load_project
from autolabel.detectors import Query, create_detector
from autolabel.imaging import load_image
from autolabel.quality import sharpness
from eval.eval_detector import AGNOSTIC, curve_for

log = logging.getLogger("eval_quality")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 1, 2, 3, 4, 6])
    ap.add_argument("--threshold", type=float, default=0.2, help="Umbral de score del detector")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_project(args.project)
    if not cfg.detector.agnostic_prompt:
        raise SystemExit("El proyecto necesita detector.agnostic_prompt")
    gold = json.loads((cfg.root / "gold.json").read_text(encoding="utf-8"))
    detector = create_detector(cfg.detector.backend, cfg.detector.model_id)
    queries = [Query(AGNOSTIC, cfg.detector.agnostic_prompt)]
    min_px = min(c.min_box_px for c in cfg.classes)
    classes = set(cfg.class_names)
    originals = {name: load_image(cfg.images_dir / name) for name in gold}

    print(f"{'sigma':>5} {'nitidez(mediana)':>17} {'precisión':>10} {'recall':>8} {'F1':>6} {'AP50':>6}")
    rows = []
    for sigma in args.sigmas:
        sharp_values, dets = [], {}
        for name, img in originals.items():
            arr = np.asarray(img)
            if sigma:
                arr = cv2.GaussianBlur(arr, (0, 0), sigma)
            sharp_values.append(sharpness(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)))
            dets[name] = detector.detect(
                Image.fromarray(arr), queries,
                box_threshold=0.1, text_threshold=0.1, nms_iou=cfg.detector.nms_iou,
            )
        curve = curve_for(cfg, gold, dets, classes, AGNOSTIC, min_px, 0.5)
        at = curve.at(args.threshold)
        row = {"sigma": sigma, "sharpness_median": float(np.median(sharp_values)),
               "precision": at["precision"], "recall": at["recall"], "f1": at["f1"], "ap50": curve.ap()}
        rows.append(row)
        print(f"{sigma:>5} {row['sharpness_median']:>17.0f} {row['precision']:>10.2f} "
              f"{row['recall']:>8.2f} {row['f1']:>6.2f} {row['ap50']:>6.2f}")

    out = cfg.out_dir / "quality_validation.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    log.info("Guardado en %s", out)


if __name__ == "__main__":
    main()
