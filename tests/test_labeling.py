import asyncio
import json
import textwrap

import pytest

from autolabel.config import load_project
from autolabel.detector_client import DetectOutput
from autolabel.detectors import Detection
from autolabel.labeling import (
    ImageLabels,
    build_queries,
    drop_tiny_boxes,
    label_images,
    load_labels,
    save_labels,
    select_images,
)

PROJECT = """
name: demo
images: ./images
classes:
  - {name: a, prompt: "thing a", min_box_px: 10}
  - {name: b, prompt: "thing b", min_box_px: 20}
detector:
  agnostic_prompt: "thing"
  box_threshold: 0.3
"""


def make_project(tmp_path, verdicts: dict[str, str], extra: str = ""):
    (tmp_path / "project.yaml").write_text(textwrap.dedent(PROJECT) + extra, encoding="utf-8")
    (tmp_path / "images").mkdir()
    (tmp_path / "out").mkdir()
    records = [{"image": n, "verdict": v, "reasons": [], "metrics": None} for n, v in verdicts.items()]
    (tmp_path / "out" / "quality.json").write_text(json.dumps({"images": records}), encoding="utf-8")
    return load_project(tmp_path)


class FakeDetector:
    """Devuelve una caja fija por imagen y cuenta las llamadas."""

    def __init__(self, box=(10.0, 10.0, 60.0, 60.0)):
        self.calls: list[str] = []
        self.box = box

    async def detect(self, path):
        self.calls.append(path.name)
        return DetectOutput(200, 100, (Detection("thing", self.box, 0.9),))


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------- selección


def test_only_images_that_passed_the_filter_are_selected(tmp_path):
    cfg = make_project(tmp_path, {"a.png": "keep", "b.png": "drop", "c.png": "unsure", "d.png": "keep"})
    assert [p.name for p in select_images(cfg)] == ["a.png", "d.png"]
    assert [p.name for p in select_images(cfg, include_unsure=True)] == ["a.png", "c.png", "d.png"]


def test_limit_sample_is_reproducible_and_sorted(tmp_path):
    cfg = make_project(tmp_path, {f"{i:02d}.png": "keep" for i in range(30)})
    first = select_images(cfg, limit=5, seed=1)
    assert first == select_images(cfg, limit=5, seed=1)
    assert first != select_images(cfg, limit=5, seed=2)
    assert first == sorted(first)
    assert len(select_images(cfg, limit=999)) == 30


def test_labeling_requires_the_quality_filter_first(tmp_path):
    cfg = make_project(tmp_path, {"a.png": "keep"})
    (tmp_path / "out" / "quality.json").unlink()
    with pytest.raises(FileNotFoundError, match="filtro de calidad"):
        select_images(cfg)


def test_build_queries(tmp_path):
    cfg = make_project(tmp_path, {})
    assert [(q.name, q.prompt) for q in build_queries(cfg, "direct")] == [("a", "thing a"), ("b", "thing b")]
    assert [(q.name, q.prompt) for q in build_queries(cfg, "agnostic")] == [("thing", "thing")]


def test_agnostic_mode_needs_a_prompt(tmp_path):
    cfg = make_project(tmp_path, {})
    no_prompt = cfg.model_copy(update={"detector": cfg.detector.model_copy(update={"agnostic_prompt": None})})
    with pytest.raises(ValueError, match="agnostic_prompt"):
        build_queries(no_prompt, "agnostic")


# ----------------------------------------------------------------------- cajas chicas


def test_tiny_boxes_use_the_class_minimum_and_the_smallest_for_the_agnostic_prompt(tmp_path):
    cfg = make_project(tmp_path, {})
    dets = (
        Detection("a", (0, 0, 12, 50), 0.9),       # lado menor 12 >= 10: queda
        Detection("b", (0, 0, 12, 50), 0.9),       # lado menor 12 < 20: se va
        Detection("thing", (0, 0, 11, 50), 0.9),   # agnóstico: mínimo del proyecto, 10: queda
        Detection("thing", (0, 0, 9, 50), 0.9),    # 9 < 10: se va
    )
    assert [d.cls for d in drop_tiny_boxes(cfg, dets)] == ["a", "thing"]
    assert drop_tiny_boxes(cfg, ()) == ()


# ----------------------------------------------------------------------- persistencia y reanudación


def test_labels_roundtrip(tmp_path):
    items = [ImageLabels("a.png", 200, 100, (Detection("x", (1.0, 2.0, 3.0, 4.0), 0.5),)),
             ImageLabels("b.png", 10, 10, ())]
    save_labels(tmp_path / "l.json", {"k": 1}, items)
    meta, loaded = load_labels(tmp_path / "l.json")
    assert meta == {"k": 1} and loaded == items
    assert not (tmp_path / "l.tmp").exists()


def test_label_images_resumes_and_restarts_when_config_changes(tmp_path):
    cfg = make_project(tmp_path, {f"{n}.png": "keep" for n in "abc"})
    paths = select_images(cfg)
    queries = build_queries(cfg, "agnostic")

    d1 = FakeDetector()
    first = run(label_images(cfg, d1, paths, queries, "agnostic"))
    assert d1.calls == ["a.png", "b.png", "c.png"]
    assert [r.image for r in first] == ["a.png", "b.png", "c.png"]

    d2 = FakeDetector()                                  # misma configuración: no repite nada
    run(label_images(cfg, d2, paths, queries, "agnostic"))
    assert d2.calls == []

    d3 = FakeDetector()                                  # solo falta una: procesa solo esa
    run(label_images(cfg, d3, [*paths[:2], tmp_path / "images" / "z.png"], queries, "agnostic"))
    assert d3.calls == ["z.png"]

    changed = cfg.model_copy(update={"detector": cfg.detector.model_copy(update={"box_threshold": 0.5})})
    d4 = FakeDetector()                                  # umbral distinto: se rehace todo
    run(label_images(changed, d4, paths, queries, "agnostic"))
    assert d4.calls == ["a.png", "b.png", "c.png"]


def test_label_images_filters_tiny_boxes(tmp_path):
    cfg = make_project(tmp_path, {"a.png": "keep"})
    result = run(label_images(cfg, FakeDetector(box=(0, 0, 4, 4)), select_images(cfg),
                              build_queries(cfg, "agnostic"), "agnostic"))
    assert result[0].detections == ()
