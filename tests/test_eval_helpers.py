import numpy as np
import pytest

from autolabel.config import ClassConfig, ProjectConfig
from autolabel.loop import BoxState, ImageState
from autolabel.verification import BoxVerdict
from eval.eval_pipeline import evaluate_stage, summarize
from eval.eval_verifier import (
    match_detections,
    precision_at_recall,
    suggest_thresholds,
)
from eval.gold import load_gold, part_of, suffix


def make_cfg(tmp_path):
    classes = tuple(ClassConfig(name=n, prompt=f"p{n}", min_box_px=10) for n in "ab")
    return ProjectConfig(name="demo", root=tmp_path, classes=classes)


# --------------------------------------------------------------------------- partición


def test_gold_partition_is_stable_disjoint_and_roughly_40_60(tmp_path):
    names = [f"img_{i}.jpg" for i in range(1000)]
    parts = [part_of(n) for n in names]
    assert parts == [part_of(n) for n in names]
    share = parts.count("calib") / len(parts)
    assert 0.35 < share < 0.45


def test_load_gold_filters_by_part(tmp_path):
    import json

    cfg = make_cfg(tmp_path)
    gold = {f"i{i}.jpg": {"size": [10, 10], "boxes": []} for i in range(50)}
    (tmp_path / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
    calib, ev, everything = load_gold(cfg, "calib"), load_gold(cfg, "eval"), load_gold(cfg, "all")
    assert set(calib) | set(ev) == set(everything) and not set(calib) & set(ev)
    assert suffix("all") == "" and suffix("calib") == "_calib"


# --------------------------------------------------------------------------- emparejamiento


def test_match_detections_marks_hits_duplicates_and_ignored_boxes():
    gt = np.array([[0, 0, 50, 50], [100, 100, 150, 150]])
    ignore = np.array([[300, 300, 305, 305]])
    dets = [
        ["x", 0, 0, 50, 50, 0.9],            # acierto
        ["x", 2, 2, 50, 50, 0.8],            # duplicado de la misma verdad: falso positivo
        ["x", 200, 0, 250, 50, 0.7],         # falso positivo
        ["x", 300, 300, 305, 305, 0.6],      # sobre una verdad ignorada: no cuenta
    ]
    flags, matched = match_detections(dets, gt, ["a", "b"], ignore)
    assert flags[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(flags[3])
    assert matched[0] == "a" and matched[1] is None


def test_precision_at_recall():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1.0, 0.0, 1.0, 1.0])
    assert precision_at_recall(scores, labels, 0.3) == pytest.approx(1.0)        # basta el primer acierto
    assert precision_at_recall(scores, labels, 0.34) == pytest.approx(2 / 3)     # ya hacen falta dos de tres
    assert precision_at_recall(scores, labels, 1.0) == pytest.approx(0.75)       # los tres aciertos entre 4


def test_suggested_thresholds_reach_the_target_precision():
    rng = np.random.default_rng(0)
    correct = rng.random(600) < 0.6
    confidence = np.clip(np.where(correct, rng.normal(0.8, 0.12, 600), rng.normal(0.4, 0.15, 600)), 0, 1)
    out = suggest_thresholds(confidence, correct)
    assert set(out["accept"]) <= {0.80, 0.85, 0.90, 0.95}
    info = out["accept"][0.95]
    accepted = confidence >= info["threshold"]
    assert correct[accepted].mean() >= 0.95
    assert info["accepted_share"] == pytest.approx(accepted.mean(), abs=0.01)
    below = confidence < out["review"]
    assert correct[below].sum() <= 0.05 * correct.sum() + 1


def test_no_threshold_when_confidence_is_uninformative():
    rng = np.random.default_rng(1)
    out = suggest_thresholds(rng.random(300), rng.random(300) < 0.5)
    assert out["accept"] == {}


# --------------------------------------------------------------------------- evaluación del flujo


def verdict(cls):
    return BoxVerdict((0, 0, 1, 1), "t", 0.5, 0.0, 1.0, 0.0, cls, {"a": 1.0}, False)


def box(b, cls="a", status="accepted", strategies=(0,)):
    return BoxState(b, 0.5, "t", list(strategies), ["m"], verdict(cls), 0.9, status)


GOLD = {
    "auto.jpg": {"size": [300, 300], "boxes": [
        {"cls": "a", "box": [0, 0, 50, 50]}, {"cls": "b", "box": [100, 100, 150, 150]},
        {"cls": "a", "box": [200, 200, 204, 204]}]},               # esta es chica y se ignora
    "review.jpg": {"size": [300, 300], "boxes": [
        {"cls": "a", "box": [0, 0, 50, 50]}, {"cls": "a", "box": [100, 100, 150, 150]}]},
    "dropped.jpg": {"size": [300, 300], "boxes": [{"cls": "a", "box": [0, 0, 50, 50]}]},
}


def test_stage_metrics_count_auto_review_and_losses(tmp_path):
    cfg = make_cfg(tmp_path)
    states = {
        "auto.jpg": ImageState("auto.jpg", 300, 300, [
            box((0, 0, 50, 50), "a"),                                  # acierto con clase correcta
            box((100, 100, 150, 150), "a"),                            # buena caja, clase equivocada
        ], "accepted"),
        "review.jpg": ImageState("review.jpg", 300, 300, [
            box((0, 0, 50, 50), "a", "uncertain"),                      # coincide con la verdad
            box((200, 0, 250, 50), "a", "uncertain"),                   # no coincide
            box((0, 200, 50, 250), "a", "rejected"),                    # rechazada: el humano no la ve
        ], "review"),
    }
    m = evaluate_stage(states, GOLD, cfg)
    assert m["total_gt"] == 5 and m["lost_by_quality"] == 1
    assert (m["auto_images"], m["review_images"]) == (1, 1)
    assert (m["auto_boxes"], m["auto_tp"], m["auto_class_ok"]) == (2, 2, 1)
    assert m["undetected_in_auto"] == 0
    assert (m["visible_boxes"], m["wrong_visible"]) == (2, 1)
    assert (m["review_gt_covered"], m["missing_in_review"]) == (1, 1)

    s = summarize(m)
    assert s["esfuerzo por imagen (todo lo visible = 1)"] == pytest.approx((2 + 1) / 5)

    # Vista por caja: lo aceptado queda confirmado y solo se revisa lo dudoso
    assert m["acc_boxes"] == 2 and m["acc_tp"] == 2 and m["acc_class_ok"] == 1
    assert m["unc_boxes"] == 2 and m["unc_tp"] == 1 and m["unc_class_ok"] == 1
    assert m["covered"] == 3                                      # 2 de la automática y 1 de la revisada
    assert s["precisión de las confirmadas (caja)"] == pytest.approx(1.0)
    assert s["precisión de las confirmadas (caja y clase)"] == pytest.approx(0.5)
    assert s["errores que quedan en lo confirmado"] == 1
    assert s["objetos que habría que dibujar"] == 2               # 5 de la verdad menos 3 cubiertos
    assert s["esfuerzo por caja (revisar = 0.25)"] == pytest.approx((0.25 * 2 + 2) / 5)
    assert s["esfuerzo por caja (revisar = 1)"] == pytest.approx((2 + 2) / 5)
