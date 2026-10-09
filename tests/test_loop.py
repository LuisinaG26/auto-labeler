import numpy as np
import pytest
from PIL import Image

from autolabel.calibration import Calibration, fit_logistic
from autolabel.config import ClassConfig, ProjectConfig, Thresholds
from autolabel.detectors import Detection
from autolabel.loop import (
    BoxState,
    ImageState,
    accepted_labels,
    classify_box,
    decide_image,
    merge_detections,
    review_labels,
    verify_pending,
)
from autolabel.verification import BoxVerdict


def make_cfg(tmp_path):
    classes = (
        ClassConfig(name="a", prompt="pa", thresholds=Thresholds(auto_accept=0.8, review=0.4)),
        ClassConfig(name="b", prompt="pb", thresholds=Thresholds(auto_accept=0.9, review=0.5)),
    )
    return ProjectConfig(name="demo", root=tmp_path, classes=classes)


def verdict(cls="a", box=(0, 0, 10, 10), score=0.5, group=False, p_one=0.9):
    return BoxVerdict(box, "thing", score, 0.05, p_one, 0.05, cls, {"a": 0.9, "b": 0.1} if cls else {}, group)


def box(conf, cls="a", b=(0, 0, 10, 10)):
    state = BoxState(b, 0.5, "thing", [0], ["m"], verdict(cls, b), conf)
    return state


# --------------------------------------------------------------------------- fusión


def test_merge_creates_new_boxes_and_records_agreement_on_matches():
    image = ImageState("x.png")
    assert merge_detections(image, [Detection("t", (0, 0, 10, 10), 0.6)], 0, "m1") == 1
    new = merge_detections(
        image, [Detection("t", (1, 1, 10, 10), 0.7), Detection("t", (50, 50, 60, 60), 0.4)], 1, "m2"
    )
    assert new == 1 and len(image.boxes) == 2
    first = image.boxes[0]
    assert first.strategies == [0, 1] and first.models == ["m1", "m2"]
    assert image.boxes[1].strategies == [1] and image.boxes[1].det_score == 0.4


def test_same_model_twice_does_not_inflate_agreement():
    image = ImageState("x.png")
    merge_detections(image, [Detection("t", (0, 0, 10, 10), 0.6)], 0, "m1")
    merge_detections(image, [Detection("t", (0, 0, 10, 10), 0.6)], 1, "m1")
    assert image.boxes[0].models == ["m1"] and image.boxes[0].strategies == [0, 1]


def test_state_roundtrips_through_plain_dicts():
    boxes = [box(0.7), BoxState((1, 2, 3, 4), 0.3, "t")]
    image = ImageState("x.png", 200, 100, boxes, "active", 1, [0.5], "")
    assert ImageState.from_dict(image.to_dict()) == image
    import json
    json.dumps(image.to_dict())                                # serializable tal cual para el checkpoint


# --------------------------------------------------------------------------- clasificación de cajas


@pytest.mark.parametrize(
    ("cls", "conf", "expected"),
    [("a", 0.85, "accepted"), ("a", 0.6, "uncertain"), ("a", 0.3, "rejected"),
     ("b", 0.85, "uncertain"),                             # los umbrales son de cada clase
     (None, 0.99, "rejected")],                            # sin clase: el VLM no vio un objeto
)
def test_classify_box_uses_the_class_thresholds(tmp_path, cls, conf, expected):
    assert classify_box(box(conf, cls), make_cfg(tmp_path)) == expected


def test_a_box_that_was_never_verified_stays_pending(tmp_path):
    assert classify_box(BoxState((0, 0, 1, 1), 0.5, "t"), make_cfg(tmp_path)) == "pending"


# --------------------------------------------------------------------------- decisión de la imagen


def image_with(*confs):
    boxes = [box(c, b=(i * 20, 0, i * 20 + 10, 10)) for i, c in enumerate(confs)]
    return ImageState("x.png", 100, 100, boxes)


def test_image_without_doubtful_boxes_is_accepted(tmp_path):
    img = image_with(0.95, 0.9, 0.1)                          # dos aceptadas y una rechazada
    assert decide_image(img, make_cfg(tmp_path), can_retry=True, min_gain=0.02) == "accepted"


def test_doubtful_boxes_keep_the_image_active_while_there_is_budget(tmp_path):
    img = image_with(0.95, 0.6)
    assert decide_image(img, make_cfg(tmp_path), can_retry=True, min_gain=0.02) == "active"
    assert img.resolved == [0.5] and img.rounds == 1


def test_doubtful_boxes_go_to_review_when_attempts_run_out(tmp_path):
    img = image_with(0.95, 0.6, 0.55)
    assert decide_image(img, make_cfg(tmp_path), can_retry=False, min_gain=0.02) == "review"
    assert img.reason == "cajas dudosas: 2"


def test_an_attempt_that_does_not_improve_stops_the_retries(tmp_path):
    cfg = make_cfg(tmp_path)
    img = image_with(0.95, 0.6)
    decide_image(img, cfg, can_retry=True, min_gain=0.02)
    assert decide_image(img, cfg, can_retry=True, min_gain=0.02) == "review"      # sigue en 0.5
    assert "sin mejora" in img.reason


def test_an_attempt_that_improves_keeps_going(tmp_path):
    cfg = make_cfg(tmp_path)
    img = image_with(0.95, 0.6, 0.6)
    decide_image(img, cfg, can_retry=True, min_gain=0.02)                          # resuelto 1/3
    img.boxes[1].confidence = 0.95
    assert decide_image(img, cfg, can_retry=True, min_gain=0.02) == "active"       # ahora 2/3


def test_an_image_with_no_boxes_keeps_trying_until_the_strategies_run_out(tmp_path):
    cfg = make_cfg(tmp_path)
    empty = ImageState("x.png")
    for _ in range(3):                         # sin cajas no hay mejora que medir, así que no se corta antes
        assert decide_image(empty, cfg, can_retry=True, min_gain=0.02) == "active"


def test_image_with_no_boxes_retries_and_then_goes_to_review(tmp_path):
    cfg = make_cfg(tmp_path)
    assert decide_image(ImageState("x.png"), cfg, can_retry=True, min_gain=0.02) == "active"
    empty = ImageState("y.png")
    assert decide_image(empty, cfg, can_retry=False, min_gain=0.02) == "review"
    assert empty.reason == "sin cajas"


# --------------------------------------------------------------------------- verificación


class FakeVerifier:
    def __init__(self):
        self.calls = []

    def verify_box(self, img, det, *, group_geometry=False):
        self.calls.append((det.box, group_geometry))
        return verdict("a", det.box, det.score, group_geometry)


def test_only_new_boxes_are_sent_to_the_vlm_and_group_flags_are_refreshed():
    img = Image.fromarray(np.zeros((120, 120, 3), np.uint8))
    state = ImageState("x.png")
    merge_detections(state, [Detection("t", (0, 0, 100, 100), 0.5)], 0, "m")
    verifier = FakeVerifier()
    assert verify_pending(state, img, verifier, None) == 1
    assert state.boxes[0].verdict.group_geometry is False
    assert state.boxes[0].confidence == 0.5                       # sin calibración: score del detector

    inner = [Detection("t", (5, 5, 40, 40), 0.5), Detection("t", (55, 55, 95, 95), 0.5)]
    merge_detections(state, inner, 1, "m2")
    assert verify_pending(state, img, verifier, None) == 2
    assert len(verifier.calls) == 3                               # la primera caja no se volvió a consultar
    assert state.boxes[0].verdict.group_geometry is True          # ahora contiene a dos
    assert not state.boxes[1].verdict.group_geometry


def test_confidence_comes_from_the_calibration_when_there_is_one():
    img = Image.fromarray(np.zeros((50, 50, 3), np.uint8))
    x = np.array([[0.1, 0.1, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0]] * 3)
    calibration = Calibration(fit_logistic(x, np.array([0.0, 1.0] * 3)))
    state = ImageState("x.png")
    merge_detections(state, [Detection("t", (0, 0, 20, 20), 0.9)], 0, "m")
    verify_pending(state, img, FakeVerifier(), calibration)
    assert 0.0 < state.boxes[0].confidence < 1.0 and state.boxes[0].confidence != 0.9


# --------------------------------------------------------------------------- salidas


def test_accepted_and_review_exports_pick_the_right_boxes(tmp_path):
    cfg = make_cfg(tmp_path)
    img = image_with(0.95, 0.6, 0.1)
    decide_image(img, cfg, can_retry=False, min_gain=0.0)
    assert [d.score for d in accepted_labels(img).detections] == [0.95]
    assert [d.score for d in review_labels(img).detections] == [0.95, 0.6]
    assert accepted_labels(img).width == 100
