"""Evalúa y calibra el verificador contra el set dorado de un proyecto.

Etapa A: clasificación. Recortes de cajas verdaderas, balanceados por clase, con y sin
         imágenes de referencia (si el proyecto las tiene). Mide cuánto acierta la clase.
Etapa B: verificación. Detecciones reales del detector, etiquetadas como acierto o falso
         positivo contra la verdad. Mide si la verificación separa unos de otros mejor que
         el score del detector solo, ajusta la confianza combinada de la caja y de la clase,
         y sugiere umbrales de aceptación con validación cruzada.

Por defecto usa la parte de calibración del set dorado (ver ``eval/gold.py``), de modo que la
parte de evaluación queda sin tocar para medir el flujo completo.

Los resultados del modelo se guardan en ``out/verifier_cache.jsonl`` para poder repetir el
análisis sin volver a consultarlo.

Uso:
    python -m eval.eval_verifier --project projects/tomatod --part calib
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autolabel.calibration import (
    CLASS_FEATURES,
    FEATURES,
    Calibration,
    class_features_of,
    features_of,
    fit_logistic,
    roc_auc,
)
from autolabel.config import ProjectConfig, load_project
from autolabel.detectors import Detection
from autolabel.geometry import box_sizes, iou_matrix
from autolabel.imaging import load_image
from autolabel.references import load_references, reference_sheet
from autolabel.verification import BoxVerdict, Verifier, contains_others
from autolabel.vlm import create_vlm
from eval.gold import load_gold, suffix

log = logging.getLogger("eval_verifier")

_MIN_CLASS_SAMPLES = 30        # mínimo de aciertos para ajustar el modelo de la clase


# --------------------------------------------------------------------------- caché


class Cache:
    """Resultados del VLM por (imagen, caja, configuración), en un archivo de líneas JSON."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self._data[rec["key"]] = rec["verdict"]

    def get(self, key: str) -> BoxVerdict | None:
        d = self._data.get(key)
        if d is None:
            return None
        return BoxVerdict(**{**d, "box": tuple(d["box"])})

    def put(self, key: str, verdict: BoxVerdict) -> None:
        self._data[key] = asdict(verdict)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "verdict": asdict(verdict)}) + "\n")


def _key(cfg: ProjectConfig, tag: str, image: str, box, min_one: float) -> str:
    # La clave incluye el modelo y el contenido de las clases: si cambian, el caché no se reutiliza
    context = json.dumps([cfg.vlm.model, [(c.name, c.definition, c.not_this) for c in cfg.classes],
                          cfg.scene, cfg.detector.agnostic_prompt, min_one], sort_keys=True)
    digest = hashlib.sha1(context.encode()).hexdigest()[:10]
    return f"{digest}|{tag}|{image}|{','.join(f'{v:.1f}' for v in box)}"


# --------------------------------------------------------------------------- etapa A


def sample_gt_crops(gold: dict, classes: tuple[str, ...], per_class: int, min_px: int, seed: int):
    rng = random.Random(seed)
    pools: dict[str, list] = {c: [] for c in classes}
    for name, entry in gold.items():
        for b in entry["boxes"]:
            if min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) >= min_px:
                pools[b["cls"]].append((name, tuple(b["box"]), b["cls"]))
    return [item for c in classes for item in rng.sample(pools[c], min(per_class, len(pools[c])))]


def run_classification(cfg, vlm, cache, crops, with_refs: bool):
    refs = load_references(cfg) if with_refs else {}
    sheet = reference_sheet(refs, list(cfg.class_names)) if refs else None
    verifier = Verifier(cfg, vlm, sheet, class_check_min_one=0.0)
    tag = "A-refs" if sheet is not None else "A-text"
    images: dict[str, object] = {}
    out = []
    for name, box, truth in crops:
        key = _key(cfg, tag, name, box, 0.0)
        verdict = cache.get(key)
        if verdict is None:
            img = images.setdefault(name, load_image(cfg.images_dir / name))
            verdict = verifier.verify_box(img, Detection("x", box, 1.0))
            cache.put(key, verdict)
        out.append((truth, verdict))
    return out


def report_classification(title: str, results, classes) -> dict:
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), int)
    for truth, v in results:
        matrix[idx[truth], idx[v.cls]] += 1
    recall = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    seen = [(t, v) for t, v in results if v.p_one >= 0.5]
    acc_seen = np.mean([v.cls == t for t, v in seen]) if seen else float("nan")
    print(f"\n-- Clasificación: {title}  (n={len(results)})")
    print(f"   exactitud {matrix.trace() / matrix.sum():.3f} | recall medio por clase {recall.mean():.3f} | "
          f"exactitud cuando ve un objeto (p_one>=0.5, n={len(seen)}) {acc_seen:.3f}")
    print("   filas = verdad, columnas = predicción:", list(classes))
    for c, row, r in zip(classes, matrix, recall, strict=True):
        print(f"   {c:<12} {row.tolist()}  recall {r:.2f}")
    return {"accuracy": float(matrix.trace() / matrix.sum()), "macro_recall": float(recall.mean()),
            "accuracy_when_object": float(acc_seen), "matrix": matrix.tolist()}


# --------------------------------------------------------------------------- etapa B


def match_detections(dets, gt_boxes, gt_classes, ignore_boxes, iou_thr=0.5):
    """Acierto o falso positivo de cada detección, y la clase de la verdad con la que coincidió.

    Emparejamiento voraz por score. Una detección sobre una verdad demasiado chica no cuenta
    (``nan``). Devuelve ``(es_acierto, clase_verdadera)``.
    """
    boxes = np.array([d[1:5] for d in dets]).reshape(-1, 4)
    scores = np.array([d[5] for d in dets])
    ious = iou_matrix(boxes, gt_boxes)
    ignored = (iou_matrix(boxes, ignore_boxes) >= iou_thr).any(axis=1)
    taken = np.zeros(len(gt_boxes), bool)
    tp = np.zeros(len(dets), bool)
    keep = np.ones(len(dets), bool)
    matched_class: list[str | None] = [None] * len(dets)
    for i in np.argsort(-scores, kind="stable"):
        if len(gt_boxes):
            candidates = np.where(taken, -1.0, ious[i])
            j = int(np.argmax(candidates))
            if candidates[j] >= iou_thr:
                taken[j], tp[i] = True, True
                matched_class[i] = gt_classes[j]
                continue
        keep[i] = not ignored[i]
    return np.where(keep, tp, np.nan), matched_class


def precision_at_recall(scores, labels, recall: float) -> float:
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    if hits[-1] == 0:
        return float("nan")
    i = int(np.searchsorted(hits, recall * hits[-1]))
    return float(hits[i] / (i + 1))


def _folds(groups, folds: int, seed: int = 0) -> np.ndarray:
    """Pliegue de cada caja. Todas las de una imagen caen en el mismo, para no filtrar información."""
    names = sorted(set(groups))
    random.Random(seed).shuffle(names)
    fold_of = {n: i % folds for i, n in enumerate(names)}
    return np.array([fold_of[g] for g in groups])


def cross_validated_scores(x, y, groups, folds=5):
    """Probabilidad por caja de un modelo ajustado sin las imágenes de su propio pliegue."""
    fold = _folds(groups, folds)
    out = np.zeros(len(y))
    for f in range(folds):
        model = fit_logistic(x[fold != f], y[fold != f])
        out[fold == f] = model.predict(x[fold == f])
    return out


def cross_validated_class_scores(xc, class_ok, is_tp, groups, folds=5):
    """Probabilidad de que la clase sea correcta. Se aprende solo de los aciertos de localización
    y se aplica a todas las cajas."""
    fold = _folds(groups, folds)
    out = np.ones(len(class_ok))
    for f in range(folds):
        train = (fold != f) & is_tp
        if train.sum() < _MIN_CLASS_SAMPLES or len(set(class_ok[train])) < 2:
            continue
        model = fit_logistic(xc[train], class_ok[train], feature_names=CLASS_FEATURES)
        out[fold == f] = model.predict(xc[fold == f])
    return out


def suggest_thresholds(confidence, correct, *, min_accepted: int = 20) -> dict:
    """Umbral de aceptación que logra cada precisión objetivo, y umbral de revisión.

    ``accept``: el menor umbral cuya precisión (entre lo aceptado) llega al objetivo.
    ``review``: por debajo de este valor hay a lo sumo un 5% de las cajas correctas.
    """
    order = np.argsort(-confidence, kind="stable")
    conf, ok = confidence[order], correct[order].astype(float)
    precision = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    out: dict = {"accept": {}}
    for target in (0.80, 0.85, 0.90, 0.95):
        valid = np.flatnonzero((precision >= target) & (np.arange(1, len(ok) + 1) >= min_accepted))
        if len(valid):
            i = int(valid[-1])
            out["accept"][target] = {
                "threshold": float(conf[i]),
                "accepted_share": (i + 1) / len(ok),
                "correct_share_accepted": float(ok[: i + 1].sum() / max(ok.sum(), 1)),
            }
    # Correctas con confianza estrictamente menor que la de cada posición: la suma del resto de la lista
    correct_below = np.cumsum(ok[::-1])[::-1] - ok
    allowed = np.flatnonzero(correct_below <= 0.05 * ok.sum())
    out["review"] = float(conf[allowed[0]]) if len(allowed) else float(conf[-1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--part", choices=["all", "calib", "eval"], default="calib")
    ap.add_argument("--per-class", type=int, default=50, help="Recortes por clase en la etapa A")
    ap.add_argument("--det-images", type=int, default=30, help="Imágenes del set dorado en la etapa B")
    ap.add_argument("--threshold", type=float, default=0.2, help="Score mínimo de las detecciones")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg = load_project(args.project)
    gold = load_gold(cfg, args.part)
    vlm = create_vlm(cfg.vlm)
    cache = Cache(cfg.out_dir / "verifier_cache.jsonl")
    min_px = min(c.min_box_px for c in cfg.classes)
    report: dict = {"model": cfg.vlm.model, "part": args.part}

    # ---- Etapa A
    crops = sample_gt_crops(gold, cfg.class_names, args.per_class, max(min_px, 20), args.seed)
    log.info("Etapa A: %d recortes verdaderos", len(crops))
    has_refs = bool(load_references(cfg))
    results_a = {}
    for with_refs in (False, True) if has_refs else (False,):
        t = time.perf_counter()
        res = run_classification(cfg, vlm, cache, crops, with_refs)
        title = "con referencias" if with_refs else "solo texto"
        results_a[title] = report_classification(title, res, cfg.class_names)
        log.info("   (%.1f s por recorte)", (time.perf_counter() - t) / len(crops))
    report["classification"] = results_a
    best_refs = has_refs and (
        results_a["con referencias"]["macro_recall"] > results_a["solo texto"]["macro_recall"]
    )

    if args.skip_b:
        _save(cfg, report, args.part)
        return

    # ---- Etapa B
    raw_path = next(cfg.out_dir.glob(f"raw_*agnostic_0.1{suffix(args.part)}.json"), None)
    if raw_path is None:
        raise SystemExit(
            f"Faltan las detecciones crudas: corre antes eval.eval_detector --part {args.part}"
        )
    raw = json.loads(raw_path.read_text(encoding="utf-8"))["dets"]
    names = sorted(gold)
    sample = random.Random(args.seed).sample(names, min(args.det_images, len(names)))
    refs = load_references(cfg) if best_refs else {}
    sheet = reference_sheet(refs, list(cfg.class_names)) if refs else None
    verifier = Verifier(cfg, vlm, sheet)
    tag = "B-refs" if sheet is not None else "B-text"
    log.info("\nEtapa B: %d imágenes, %s", len(sample), "con referencias" if sheet else "solo texto")

    rows = []                      # (imagen, verdict, es_acierto, clase_verdadera)
    t = time.perf_counter()
    for k, name in enumerate(sample, 1):
        entry = gold[name]
        gt = np.array([b["box"] for b in entry["boxes"]]).reshape(-1, 4)
        gt_cls = [b["cls"] for b in entry["boxes"]]
        small = box_sizes(gt).min(axis=1) < min_px if len(gt) else np.zeros(0, bool)
        dets = [d for d in raw[name] if d[5] >= args.threshold]
        if not dets:
            continue
        keep = [c for c, s in zip(gt_cls, small, strict=True) if not s]
        flags, matched = match_detections(dets, gt[~small], keep, gt[small])
        img = load_image(cfg.images_dir / name)
        groups = contains_others([d[1:5] for d in dets], classes=[d[0] for d in dets]) >= 2
        for d, is_tp, truth, grp in zip(dets, flags, matched, groups, strict=True):
            if np.isnan(is_tp):
                continue
            key = _key(cfg, tag, name, d[1:5], 0.2)
            verdict = cache.get(key)
            if verdict is None:
                det = Detection(d[0], tuple(d[1:5]), d[5])
                verdict = verifier.verify_box(img, det, group_geometry=bool(grp))
                cache.put(key, verdict)
            rows.append((name, verdict, bool(is_tp), truth))
        if k % 5 == 0:
            log.info("  %d/%d imágenes, %d cajas, %.1f s por caja", k, len(sample), len(rows),
                     (time.perf_counter() - t) / max(1, len(rows)))

    y = np.array([r[2] for r in rows], dtype=float)
    x = np.array([features_of(r[1]) for r in rows])
    xc = np.array([class_features_of(r[1]) for r in rows])
    groups = [r[0] for r in rows]
    labels_b = y.astype(bool)
    class_ok = np.array([r[2] and r[1].cls == r[3] for r in rows])
    det_score, p_one = x[:, 0], x[:, 1]
    box_cv = cross_validated_scores(x, y, groups)

    print(f"\n-- Verificación de detecciones (n={len(rows)} cajas, {int(y.sum())} aciertos, "
          f"{len(rows) - int(y.sum())} falsos positivos)")
    print(f"   {'señal':<30} {'AUC':>6} {'prec@rec50':>11} {'prec@rec80':>11}")
    signals = {"score del detector": det_score, "p(un objeto) del VLM": p_one,
               "detector x VLM": det_score * p_one, "caja combinada (validación cruzada)": box_cv}
    report["detections"] = {}
    for label, s in signals.items():
        auc = roc_auc(s, labels_b)
        p50, p80 = precision_at_recall(s, y, 0.5), precision_at_recall(s, y, 0.8)
        print(f"   {label:<36} {auc:>6.3f} {p50:>11.3f} {p80:>11.3f}")
        report["detections"][label] = {"auc": auc, "prec_at_rec50": p50, "prec_at_rec80": p80}

    geo = x[:, 3] == 1
    if geo.any():
        print(f"\n   cajas de grupo por geometría: {int(geo.sum())} ({100 * geo.mean():.1f}%), "
              f"aciertos entre ellas {y[geo].mean():.2f} (frente a {y[~geo].mean():.2f} en las demás)")

    # ---- Modelos finales: la caja y, si hay varias clases, la clase
    box_model = fit_logistic(x, y)
    class_model = None
    multi_class = len(cfg.classes) > 1
    tp_with_class = labels_b & np.array([bool(r[1].class_probs) for r in rows])
    if multi_class and tp_with_class.sum() >= _MIN_CLASS_SAMPLES and len(set(class_ok[tp_with_class])) == 2:
        class_model = fit_logistic(xc[tp_with_class], class_ok[tp_with_class].astype(float),
                                   feature_names=CLASS_FEATURES)
        class_acc = class_ok[tp_with_class].mean()
        print(f"\n   clase correcta entre las cajas bien localizadas: {class_acc:.2f} "
              f"(n={int(tp_with_class.sum())})")
    Calibration(box_model, class_model).save(cfg.out_dir / "calibration.json")
    print("\n   pesos de la confianza de la caja (variables estandarizadas):")
    for f, w in zip(FEATURES, box_model.weights, strict=True):
        print(f"     {f:<16} {w:+.2f}")
    if class_model:
        print("   pesos de la confianza de la clase:")
        for f, w in zip(CLASS_FEATURES, class_model.weights, strict=True):
            print(f"     {f:<16} {w:+.2f}")

    # ---- Umbrales sugeridos, con predicciones que no vieron su propia imagen
    class_cv = cross_validated_class_scores(xc, class_ok.astype(float), tp_with_class, groups) \
        if multi_class else np.ones(len(rows))
    final_cv = box_cv * class_cv
    thresholds = suggest_thresholds(final_cv, class_ok if multi_class else labels_b)
    print(f"\n   Umbrales sugeridos para thresholds.auto_accept y thresholds.review "
          f"(sobre caja y clase correctas, n={len(rows)})")
    for target, info in thresholds["accept"].items():
        print(f"     precisión {target:.2f}: auto_accept >= {info['threshold']:.2f}  "
              f"acepta {100 * info['accepted_share']:.0f}% de las cajas "
              f"({100 * info['correct_share_accepted']:.0f}% de las correctas)")
    if not thresholds["accept"]:
        print("     ninguno alcanza una precisión de 0.90: la confianza no alcanza para aceptar sin revisar")
    print(f"     review >= {thresholds['review']:.2f} (por debajo hay como mucho 5% de las correctas)")

    report["thresholds"] = thresholds
    report["class_counts"] = dict(Counter(r[1].cls for r in rows))
    _save(cfg, report, args.part)


def _save(cfg: ProjectConfig, report: dict, part: str) -> None:
    out = cfg.out_dir / f"verifier_report{suffix(part)}.json"
    out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    log.info("\nReporte guardado en %s", out)


if __name__ == "__main__":
    main()
