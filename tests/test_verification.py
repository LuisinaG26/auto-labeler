import numpy as np
import pytest
from PIL import Image

from autolabel.config import ClassConfig, ProjectConfig
from autolabel.detectors import Detection
from autolabel.references import reference_sheet
from autolabel.verification import (
    Verifier,
    contains_others,
    crop_candidate,
    object_noun,
)


def make_cfg(tmp_path, classes=2, agnostic="thing", scene="una escena de prueba"):
    cls = tuple(
        ClassConfig(name=f"c{i}", prompt=f"prompt {i}", definition=f"def {i}", not_this=("otro",))
        for i in range(classes)
    )
    return ProjectConfig(name="demo", root=tmp_path, classes=cls, scene=scene,
                         detector={"agnostic_prompt": agnostic})


class FakeVlm:
    """Responde según el contenido del prompt y registra cada llamada."""

    def __init__(self, object_probs=(0.05, 0.9, 0.05), class_probs=None):
        self.object_probs = object_probs
        self.class_probs = class_probs
        self.calls = []

    def choose(self, images, prompt, options, *, system=None):
        self.calls.append((len(images), prompt, tuple(options)))
        if any(o.startswith("Exactly one") for o in options):
            probs = self.object_probs
        else:
            probs = self.class_probs or [1 / len(options)] * len(options)
        return dict(zip(options, probs, strict=True))

    def ask_json(self, images, prompt, schema, *, system=None):
        raise NotImplementedError


def image(w=200, h=100):
    return Image.fromarray(np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- recorte


def test_crop_adds_context_and_stays_inside_the_image():
    crop = crop_candidate(image(), (0, 0, 40, 40), min_side=10, max_side=1000)
    assert crop.width >= 40 and crop.height >= 40
    assert crop.width <= 200 and crop.height <= 100


def test_small_crops_are_enlarged_and_big_ones_capped():
    small = crop_candidate(image(), (50, 50, 60, 60), min_side=224, max_side=512)
    assert min(small.size) >= 200
    big = crop_candidate(image(2000, 2000), (0, 0, 1900, 1900), min_side=224, max_side=512)
    assert max(big.size) <= 512


# --------------------------------------------------------------------------- geometría


def test_contains_others_detects_a_group_box():
    boxes = [(0, 0, 100, 100), (5, 5, 40, 40), (55, 55, 95, 95), (200, 200, 220, 220)]
    assert contains_others(boxes).tolist() == [2, 0, 0, 0]


def test_nested_boxes_of_another_class_are_not_a_group():
    boxes = [(0, 0, 100, 100), (5, 5, 40, 40), (55, 55, 95, 95)]
    assert contains_others(boxes, classes=["camion", "rueda", "rueda"]).tolist() == [0, 0, 0]
    assert contains_others(boxes, classes=["rueda", "rueda", "rueda"]).tolist() == [2, 0, 0]
    assert contains_others(boxes, classes=["rueda", "otra", "rueda"]).tolist() == [1, 0, 0]


def test_verify_image_ignores_nesting_across_classes(tmp_path):
    cfg = make_cfg(tmp_path)
    dets = [Detection("big", (0, 0, 100, 100), 0.5), Detection("small", (5, 5, 40, 40), 0.5),
            Detection("small", (55, 55, 95, 95), 0.5)]
    verdicts = Verifier(cfg, FakeVlm()).verify_image(image(), dets)
    assert [v.group_geometry for v in verdicts] == [False, False, False]


def test_a_box_with_a_single_inner_box_is_not_a_group_and_empty_input_is_ok():
    assert contains_others([(0, 0, 100, 100), (10, 10, 50, 50)]).tolist() == [1, 0]
    assert contains_others([]).tolist() == []


# --------------------------------------------------------------------------- verificador


def test_object_noun_uses_the_agnostic_prompt_or_the_classes(tmp_path):
    assert object_noun(make_cfg(tmp_path)) == "thing"
    assert object_noun(make_cfg(tmp_path, agnostic=None)) == "prompt 0 or prompt 1"


def test_verdict_has_object_and_class_probabilities(tmp_path):
    cfg = make_cfg(tmp_path)
    vlm = FakeVlm(object_probs=(0.1, 0.7, 0.2), class_probs=[0.2, 0.8])
    v = Verifier(cfg, vlm).verify_box(image(), Detection("thing", (20, 20, 80, 80), 0.6))
    assert (v.p_none, v.p_one, v.p_several) == pytest.approx((0.1, 0.7, 0.2))
    assert v.cls == "c1" and v.class_probs == pytest.approx({"c0": 0.2, "c1": 0.8})
    assert v.det_score == 0.6 and v.detector_class == "thing"
    assert len(vlm.calls) == 2


def test_class_check_is_skipped_when_it_is_not_an_object(tmp_path):
    vlm = FakeVlm(object_probs=(0.95, 0.03, 0.02))
    v = Verifier(make_cfg(tmp_path), vlm).verify_box(image(), Detection("thing", (20, 20, 80, 80), 0.3))
    assert v.cls is None and v.class_probs == {}
    assert len(vlm.calls) == 1


def test_single_class_projects_skip_the_class_question(tmp_path):
    vlm = FakeVlm()
    verifier = Verifier(make_cfg(tmp_path, classes=1), vlm)
    v = verifier.verify_box(image(), Detection("thing", (20, 20, 80, 80), 0.5))
    assert v.cls == "c0" and v.class_probs == {"c0": 1.0}
    assert len(vlm.calls) == 1


def test_prompts_carry_the_project_text_and_no_class_is_hardcoded(tmp_path):
    cfg = make_cfg(tmp_path)
    vlm = FakeVlm()
    Verifier(cfg, vlm).verify_box(image(), Detection("thing", (20, 20, 80, 80), 0.5))
    object_prompt, class_prompt = vlm.calls[0][1], vlm.calls[1][1]
    assert "una escena de prueba" in object_prompt and "thing" in object_prompt
    assert "- c0: def 0 (not: otro)" in class_prompt and "- c1: def 1" in class_prompt


def test_reference_sheet_is_passed_first_and_mentioned(tmp_path):
    cfg = make_cfg(tmp_path)
    refs = {"c0": [Image.new("RGB", (30, 30), "red")], "c1": [Image.new("RGB", (30, 30), "green")]}
    sheet = reference_sheet(refs, ["c0", "c1"])
    vlm = FakeVlm()
    Verifier(cfg, vlm, sheet).verify_box(image(), Detection("thing", (20, 20, 80, 80), 0.5))
    n_images, prompt, _ = vlm.calls[1]
    assert n_images == 2 and "reference sheet" in prompt
    assert vlm.calls[0][0] == 1                      # la pregunta del objeto no usa la hoja


def test_verify_image_flags_group_geometry(tmp_path):
    cfg = make_cfg(tmp_path)
    dets = [Detection("thing", b, 0.5) for b in [(0, 0, 100, 100), (5, 5, 40, 40), (55, 55, 95, 95)]]
    verdicts = Verifier(cfg, FakeVlm()).verify_image(image(), dets)
    assert [v.group_geometry for v in verdicts] == [True, False, False]


def test_reference_sheet_without_references_is_none():
    assert reference_sheet({}, ["a"]) is None
