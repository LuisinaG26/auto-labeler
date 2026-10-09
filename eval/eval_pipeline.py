"""Mide el flujo completo (``autolabel run``) contra la verdad del set dorado.

Lee las fotos de cada ronda (``out/run/round_<k>.json``) y el resultado final, y calcula para
cada una qué se aceptó solo, con qué precisión, y cuánto trabajo queda para el humano. La ronda
1 equivale a etiquetar con un único intento, así que la diferencia con el final es lo que aporta
el bucle.

Dos advertencias sobre la lectura:
  * "Esfuerzo" cuenta acciones sin ponderar: confirmar o borrar una caja, o dibujar una que
    falta. Dibujar es más caro que confirmar, así que el ahorro real depende de ese cociente.
  * Las imágenes apartadas por el filtro de calidad cuentan como objetos perdidos.

Uso:
    python -m eval.eval_pipeline --project projects/tomatod --part eval
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from autolabel.config import ProjectConfig, load_project
from autolabel.geometry import box_sizes
from autolabel.loop import ImageState
from eval.eval_verifier import match_detections
from eval.gold import load_gold

log = logging.getLogger("eval_pipeline")


def _truth(entry: dict, min_px: int):
    gt = np.array([b["box"] for b in entry["boxes"]]).reshape(-1, 4)
    classes = [b["cls"] for b in entry["boxes"]]
    small = box_sizes(gt).min(axis=1) < min_px if len(gt) else np.zeros(0, bool)
    return gt[~small], [c for c, s in zip(classes, small, strict=True) if not s], gt[small]


def _match(boxes, gt, gt_classes, ignore):
    """(es_acierto, clase_correcta) por caja, sobre las que no caen en una verdad ignorada."""
    if not boxes:
        return np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, bool)
    dets = [[b.cls or "?", *b.box, b.confidence or b.det_score] for b in boxes]
    flags, matched = match_detections(dets, gt, gt_classes, ignore)
    used = ~np.isnan(flags)
    tp = np.where(used, flags, 0).astype(bool)
    class_ok = np.array([bool(t) and b.cls == m for t, b, m in zip(tp, boxes, matched, strict=True)])
    return tp, class_ok, used


def evaluate_stage(states: dict[str, ImageState], gold: dict, cfg: ProjectConfig) -> dict:
    min_px = min(c.min_box_px for c in cfg.classes)
    total_gt = sum(len(_truth(e, min_px)[0]) for e in gold.values())
    out = dict.fromkeys((
        "auto_images", "review_images", "auto_boxes", "auto_tp", "auto_class_ok", "auto_gt_found",
        "visible_boxes", "review_gt", "review_gt_covered", "missing_in_review",
        "undetected_in_auto", "lost_by_quality", "wrong_visible",
        "acc_boxes", "acc_tp", "acc_class_ok", "unc_boxes", "unc_tp", "unc_class_ok", "covered"), 0)
    out["total_gt"] = total_gt

    for name, entry in gold.items():
        gt, gt_classes, ignore = _truth(entry, min_px)
        state = states.get(name)
        if state is None:                                    # apartada por el filtro de calidad
            out["lost_by_quality"] += len(gt)
            continue

        # Vista por caja, sin importar el estado de la imagen: lo aceptado queda confirmado y solo
        # lo dudoso se revisa. Lo que ninguna de las dos cubre hay que dibujarlo.
        visible_all = [b for b in state.boxes if b.status in ("accepted", "uncertain")]
        tp_all, ok_all, used_all = _match(visible_all, gt, gt_classes, ignore)
        for b, hit, class_ok, used in zip(visible_all, tp_all, ok_all, used_all, strict=True):
            if used:
                prefix = "acc" if b.status == "accepted" else "unc"
                out[f"{prefix}_boxes"] += 1
                out[f"{prefix}_tp"] += int(hit)
                out[f"{prefix}_class_ok"] += int(class_ok)
        out["covered"] += int(tp_all.sum())

        done = state.status == "accepted"
        # Antes del final, una imagen que sigue activa o en revisión cuenta como pendiente para el humano
        if done:
            out["auto_images"] += 1
            auto = [b for b in state.boxes if b.status == "accepted"]
            tp, class_ok, used = _match(auto, gt, gt_classes, ignore)
            out["auto_boxes"] += int(used.sum())
            out["auto_tp"] += int((tp & used).sum())
            out["auto_class_ok"] += int((class_ok & used).sum())
            out["auto_gt_found"] += int(class_ok.sum())
            out["undetected_in_auto"] += len(gt) - int(tp.sum())
        else:
            out["review_images"] += 1
            visible = [b for b in state.boxes if b.status in ("accepted", "uncertain")]
            tp, _, used = _match(visible, gt, gt_classes, ignore)
            out["visible_boxes"] += int(used.sum())
            out["wrong_visible"] += int((used & ~tp).sum())
            out["review_gt"] += len(gt)
            out["review_gt_covered"] += int(tp.sum())
            out["missing_in_review"] += len(gt) - int(tp.sum())
    return out


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def summarize(m: dict, check_cost: float = 0.25) -> dict:
    """Resumen de una etapa. ``check_cost`` es lo que cuesta confirmar o borrar una caja ya
    dibujada, en unidades de lo que cuesta dibujar una: es una suposición, no una medición."""
    to_draw = m["total_gt"] - m["covered"]
    legacy_actions = m["visible_boxes"] + m["missing_in_review"]
    return {
        # Vista por caja: lo confirmado no se revisa y solo se mira lo dudoso
        "cajas confirmadas (aceptadas)": m["acc_boxes"],
        "precisión de las confirmadas (caja)": _ratio(m["acc_tp"], m["acc_boxes"]),
        "precisión de las confirmadas (caja y clase)": _ratio(m["acc_class_ok"], m["acc_boxes"]),
        "verdad resuelta por las confirmadas": _ratio(m["acc_class_ok"], m["total_gt"]),
        "errores que quedan en lo confirmado": m["acc_boxes"] - m["acc_class_ok"],
        "cajas dudosas a revisar": m["unc_boxes"],
        "acierto de las dudosas (caja y clase)": _ratio(m["unc_class_ok"], m["unc_boxes"]),
        "objetos que habría que dibujar": to_draw,
        f"esfuerzo por caja (revisar = {check_cost:g})": _ratio(
            check_cost * m["unc_boxes"] + to_draw, m["total_gt"]
        ),
        "esfuerzo por caja (revisar = 1)": _ratio(m["unc_boxes"] + to_draw, m["total_gt"]),
        # Vista por imagen, la anterior: una imagen solo se acepta si ninguna caja es dudosa
        "imágenes aceptadas enteras": m["auto_images"],
        "imágenes a revisión": m["review_images"],
        "perdidos por el filtro": m["lost_by_quality"],
        "esfuerzo por imagen (todo lo visible = 1)": _ratio(legacy_actions, m["total_gt"]),
    }


def first_strategy_share(final_states: dict[str, ImageState], gold: dict, cfg: ProjectConfig) -> dict:
    """De los objetos bien localizados, cuántos ya los encontró la primera estrategia."""
    min_px = min(c.min_box_px for c in cfg.classes)
    only_first = later = 0
    for name, entry in gold.items():
        state = final_states.get(name)
        if state is None or not state.boxes:
            continue
        gt, gt_classes, ignore = _truth(entry, min_px)
        tp, _, used = _match(state.boxes, gt, gt_classes, ignore)
        for box, hit, ok in zip(state.boxes, tp, used, strict=True):
            if ok and hit:
                if min(box.strategies) == 0:
                    only_first += 1
                else:
                    later += 1
    return {"found_by_first": only_first, "found_only_later": later}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--part", choices=["all", "calib", "eval"], default="eval")
    ap.add_argument("--run-dir", default="run", help="Carpeta de resultados dentro de out/")
    ap.add_argument("--check-cost", type=float, default=0.25,
                    help="Costo de confirmar o borrar una caja, en unidades del costo de dibujarla")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_project(args.project)
    gold = load_gold(cfg, args.part)
    run_dir = cfg.out_dir / args.run_dir
    final_data = json.loads((run_dir / "final.json").read_text(encoding="utf-8"))
    final_states = {d["name"]: ImageState.from_dict(d) for d in final_data["images"]}

    stages: dict[str, dict[str, ImageState]] = {}
    for path in sorted(run_dir.glob("round_*.json")):
        k = int(path.stem.split("_")[1])
        raw = json.loads(path.read_text(encoding="utf-8"))
        stages[f"ronda {k + 1}"] = {n: ImageState.from_dict(d) for n, d in raw.items()}
    stages["final"] = final_states

    report = {name: summarize(evaluate_stage(states, gold, cfg), args.check_cost)
              for name, states in stages.items()}
    labels = list(next(iter(report.values())))
    width = max(len(label) for label in labels)
    print(f"\nImágenes evaluadas: {len(gold)} | objetos de la verdad: "
          f"{sum(len(_truth(e, min(c.min_box_px for c in cfg.classes))[0]) for e in gold.values())}")
    print(f"{'':<{width}} " + " ".join(f"{name:>10}" for name in report))
    for label in labels:
        cells = []
        for name in report:
            v = report[name][label]
            cells.append(f"{v:>10.3f}" if isinstance(v, float) else f"{v:>10}")
        print(f"{label:<{width}} " + " ".join(cells))

    share = first_strategy_share(final_states, gold, cfg)
    print(f"\nObjetos bien localizados al final: {share['found_by_first']} ya los vio la primera "
          f"estrategia y {share['found_only_later']} solo los aportaron las siguientes")
    (run_dir / "pipeline_report.json").write_text(
        json.dumps({"stages": report, "first_strategy": share}, indent=1, default=float), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
