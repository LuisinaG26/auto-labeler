import numpy as np
from PIL import Image

from autolabel.calibration import Calibration, fit_logistic
from autolabel.config import ClassConfig, ProjectConfig
from autolabel.detectors import Detection
from autolabel.labeling import ImageLabels
from autolabel.verifying import load_verification, verify_labels


class CountingVlm:
    def __init__(self):
        self.calls = 0

    def choose(self, images, prompt, options, *, system=None):
        self.calls += 1
        probs = [0.1, 0.8, 0.1] if len(options) == 3 and options[1].startswith("Exactly") else [0.7, 0.3]
        return dict(zip(options, probs, strict=True))


def setup(tmp_path):
    classes = (ClassConfig(name="a", prompt="pa"), ClassConfig(name="b", prompt="pb"))
    cfg = ProjectConfig(name="demo", root=tmp_path, classes=classes, detector={"agnostic_prompt": "thing"})
    cfg.images_dir.mkdir()
    cfg.out_dir.mkdir()
    for n in ("1.png", "2.png"):
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 120, 3), dtype=np.uint8)).save(
            cfg.images_dir / n
        )
    det = Detection("thing", (10, 10, 60, 60), 0.5)
    labels = [ImageLabels(n, 120, 80, (det,)) for n in ("1.png", "2.png")]
    return cfg, labels


def test_verification_is_saved_and_resumed_without_new_calls(tmp_path):
    cfg, labels = setup(tmp_path)
    vlm = CountingVlm()
    first = verify_labels(cfg, vlm, labels, calibration=None)
    assert [r.image for r in first] == ["1.png", "2.png"]
    assert first[0].boxes[0].confidence == 0.5          # sin calibración: el score del detector
    assert first[0].boxes[0].cls == "a"
    calls = vlm.calls

    again = verify_labels(cfg, vlm, labels, calibration=None)
    assert vlm.calls == calls                           # nada nuevo que consultar
    assert again == first

    meta, loaded = load_verification(cfg.out_dir / "verification.json")
    assert meta["calibrated"] is False and loaded == first


def test_changing_the_calibration_invalidates_the_saved_run(tmp_path):
    cfg, labels = setup(tmp_path)
    vlm = CountingVlm()
    verify_labels(cfg, vlm, labels, calibration=None)
    before = vlm.calls
    model = Calibration(fit_logistic(np.array([[0.1, 0.1, 0.0, 0.0], [0.9, 0.9, 0.0, 0.0]] * 3),
                                     np.array([0.0, 1.0] * 3)))
    result = verify_labels(cfg, vlm, labels, calibration=model)
    assert vlm.calls > before
    assert 0.0 < result[0].boxes[0].confidence < 1.0


def test_images_without_detections_are_kept_with_no_boxes(tmp_path):
    cfg, _ = setup(tmp_path)
    result = verify_labels(cfg, CountingVlm(), [ImageLabels("1.png", 120, 80, ())], calibration=None)
    assert result[0].boxes == ()
