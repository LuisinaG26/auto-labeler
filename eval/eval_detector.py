"""Evalúa backends de detección contra el set dorado de un proyecto (fase 0).

Compara dos formas de resolver las clases:
  * direct:   un prompt por clase.
  * agnostic: un único prompt que agrupa todas las clases (mide solo localización).

Las detecciones crudas se guardan con umbrales bajos, así el barrido de umbrales
se hace sin volver a correr el modelo.

Uso:
    python -m eval.eval_detector --project projects/tomates-gold \\
        --models IDEA-Research/grounding-dino-tiny IDEA-Research/grounding-dino-base
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from autolabel.config import ProjectConfig, load_project
from autolabel.detectors import Detection, Detector, Query, create_detector
from autolabel.geometry import as_boxes, box_sizes
from autolabel.imaging import load_image
from eval.gold import load_gold, suffix
from eval.metrics import Curve, build_curve

log = logging.getLogger("eval_detector")

AGNOSTIC = "__agnostic__"


# --------------------------------------------------------------------------- detección


def run_detection(detector: Detector, cfg: ProjectConfig, gold: dict, queries: list[Query],
                  low: float, nms_iou: float, tiles: int = 1) -> tuple[dict[str, list[Detection]], float]:
    """Devuelve las detecciones por imagen y los segundos por imagen."""
    out: dict[str, list[Detection]] = {}
    options = {"box_threshold": low, "text_threshold": low, "nms_iou": nms_iou, "tiles": tiles}
    # La primera pasada calienta CUDA y cargas perezosas: no entra en el tiempo medido
    detector.detect(load_image(cfg.images_dir / next(iter(gold))), queries, **options)

    start = time.perf_counter()
    for name in gold:
        out[name] = detector.detect(load_image(cfg.images_dir / name), queries, **options)
    return out, (time.perf_counter() - start) / len(gold)


def load_or_run(get_detector: Callable[[], Detector], cfg: ProjectConfig, gold: dict, model_id: str,
                mode: str, queries: list[Query], low: float, force: bool,
                part_suffix: str = "", tiles: int = 1) -> tuple[dict[str, list[Detection]], float | None]:
    """Usa el caché si existe. El modelo solo se carga si hace falta correr."""
    slug = model_id.replace("/", "__")
    tiles_tag = f"_t{tiles}" if tiles > 1 else ""
    cache = cfg.out_dir / f"raw_{slug}_{mode}_{low}{tiles_tag}{part_suffix}.json"
    # El caché se invalida si cambian el set dorado, los prompts, el umbral de NMS o el mosaico
    fingerprint = hashlib.sha1(json.dumps(
        [gold, [(q.name, q.prompt) for q in queries], cfg.detector.nms_iou, tiles], sort_keys=True
    ).encode()).hexdigest()
    if cache.exists() and not force:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("fingerprint") == fingerprint:
            dets = {k: [Detection(d[0], tuple(d[1:5]), d[5]) for d in v] for k, v in data["dets"].items()}
            return dets, data.get("sec_per_image")
        log.info("Caché desactualizado, se vuelve a detectar (%s)", cache.name)

    dets, sec = run_detection(get_detector(), cfg, gold, queries, low, cfg.detector.nms_iou, tiles)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "model_id": model_id, "mode": mode, "low_threshold": low, "sec_per_image": sec,
        "fingerprint": fingerprint,
        "dets": {k: [[d.cls, *d.box, d.score] for d in v] for k, v in dets.items()},
    }), encoding="utf-8")
    return dets, sec


# --------------------------------------------------------------------------- evaluación


def _split_by_size(boxes: np.ndarray, min_px: int) -> tuple[np.ndarray, np.ndarray]:
    """Separa en (válidas, ignoradas) según el lado menor."""
    if len(boxes) == 0:
        return boxes, boxes
    small = box_sizes(boxes).min(axis=1) < min_px
    return boxes[~small], boxes[small]


def curve_for(cfg: ProjectConfig, gold: dict, dets: dict[str, list[Detection]],
              gt_classes: set[str], det_class: str, min_px: int, iou_thr: float) -> Curve:
    def per_image():
        for name, entry in gold.items():
            gt = as_boxes([b["box"] for b in entry["boxes"] if b["cls"] in gt_classes])
            gt, ignore = _split_by_size(gt, min_px)
            d = [x for x in dets.get(name, []) if x.cls == det_class]
            d_boxes = as_boxes([x.box for x in d])
            d_scores = np.array([x.score for x in d])
            if len(d_boxes):
                keep = box_sizes(d_boxes).min(axis=1) >= min_px
                d_boxes, d_scores = d_boxes[keep], d_scores[keep]
            yield d_boxes, d_scores, gt, ignore

    return build_curve(per_image(), iou_thr)


def summarize(curve: Curve, op_threshold: float) -> dict[str, float | int]:
    best = curve.best_f1()
    op = curve.at(op_threshold)
    return {
        "n_gt": curve.n_gt,
        "ap50": round(curve.ap(), 4),
        "best_f1": round(best["f1"], 4), "best_thr": round(best["threshold"], 3),
        "best_p": round(best["precision"], 4), "best_r": round(best["recall"], 4),
        f"p@{op_threshold}": round(op["precision"], 4), f"r@{op_threshold}": round(op["recall"], 4),
    }


def evaluate(cfg: ProjectConfig, gold: dict, direct: dict, agnostic: dict | None, iou_thr: float) -> dict:
    op = cfg.detector.box_threshold
    report: dict[str, dict] = {"direct": {}, "agnostic": {}}
    for c in cfg.classes:
        curve = curve_for(cfg, gold, direct, {c.name}, c.name, c.min_box_px, iou_thr)
        report["direct"][c.name] = summarize(curve, op)
    aps = [v["ap50"] for v in report["direct"].values()]
    report["direct"]["__mean__"] = {"ap50": round(float(np.mean(aps)), 4)}

    if agnostic is not None:
        min_px = min(c.min_box_px for c in cfg.classes)
        curve = curve_for(cfg, gold, agnostic, set(cfg.class_names), AGNOSTIC, min_px, iou_thr)
        report["agnostic"]["all"] = summarize(curve, op)
    return report


def print_report(model_id: str, sec: dict[str, float | None], report: dict) -> None:
    print(f"\n=== {model_id} ===")
    for mode in ("direct", "agnostic"):
        if not report[mode]:
            continue
        s = sec.get(mode)
        print(f"-- {mode}" + (f"  ({s * 1000:.0f} ms/imagen)" if s else ""))
        for name, m in report[mode].items():
            if name == "__mean__":
                print(f"   {'mAP50':<12} {m['ap50']:.3f}")
                continue
            print(
                f"   {name:<12} nGT={m['n_gt']:<5} AP50={m['ap50']:.3f}  mejorF1={m['best_f1']:.3f} "
                f"(thr {m['best_thr']:.2f}, P {m['best_p']:.2f}, R {m['best_r']:.2f})"
            )


# --------------------------------------------------------------------------- CLI


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--models", nargs="+", help="IDs de modelo; por defecto el del proyecto")
    ap.add_argument("--low", type=float, default=0.1, help="Umbral bajo para guardar detecciones crudas")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--force", action="store_true", help="Ignora el caché de detecciones")
    ap.add_argument("--part", choices=["all", "calib", "eval"], default="all",
                    help="Parte del set dorado a usar (ver eval/gold.py)")
    ap.add_argument("--tiles", type=int, default=1, help="Mosaico de tiles x tiles, además de la imagen")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_project(args.project)
    gold = load_gold(cfg, args.part)
    models = args.models or [cfg.detector.model_id]

    direct_q = [Query(c.name, c.prompt) for c in cfg.classes]
    agnostic_q = [Query(AGNOSTIC, cfg.detector.agnostic_prompt)] if cfg.detector.agnostic_prompt else []

    full: dict[str, dict] = {}
    for model_id in models:
        log.info("Modelo %s: %d imágenes", model_id, len(gold))
        get_detector = functools.cache(lambda m=model_id: create_detector(cfg.detector.backend, m))
        direct, s_direct = load_or_run(get_detector, cfg, gold, model_id, "direct", direct_q,
                                       args.low, args.force, suffix(args.part), args.tiles)
        agnostic, s_agn = (load_or_run(get_detector, cfg, gold, model_id, "agnostic", agnostic_q,
                                       args.low, args.force, suffix(args.part), args.tiles)
                           if agnostic_q else (None, None))
        del get_detector   # libera el modelo antes de cargar el siguiente
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report = evaluate(cfg, gold, direct, agnostic, args.iou)
        print_report(model_id, {"direct": s_direct, "agnostic": s_agn}, report)
        full[model_id] = report

    tiles_tag = f"_t{args.tiles}" if args.tiles > 1 else ""
    out = cfg.out_dir / f"phase0_report{tiles_tag}{suffix(args.part)}.json"
    out.write_text(json.dumps(full, indent=1), encoding="utf-8")
    log.info("\nReporte guardado en %s", Path(out))


if __name__ == "__main__":
    main()
