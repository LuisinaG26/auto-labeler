import json
import textwrap

import numpy as np
from PIL import Image

from autolabel.config import load_project
from autolabel.curation import Curator, curate_images, load_curation, save_curation
from autolabel.labeling import select_images


class ScriptedVlm:
    def __init__(self, probs):
        self.probs = probs
        self.json_calls = []

    def choose(self, images, prompt, options, *, system=None):
        self.prompt = prompt
        return dict(zip(options, self.probs, strict=True))

    def ask_json(self, images, prompt, schema, *, system=None):
        self.json_calls.append(prompt)
        return {"reason": "demasiado borrosa"}


def make_cfg(tmp_path, curator_line=""):
    (tmp_path / "project.yaml").write_text(textwrap.dedent(f"""
        name: demo
        scene: una escena
        classes:
          - {{name: a, prompt: "cosa a"}}
          - {{name: b, prompt: "cosa b"}}
        {curator_line}
    """), encoding="utf-8")
    (tmp_path / "images").mkdir()
    (tmp_path / "out").mkdir()
    return load_project(tmp_path)


def test_verdict_is_the_most_probable_option_and_reason_only_when_not_kept(tmp_path):
    cfg = make_cfg(tmp_path)
    keep = ScriptedVlm([0.8, 0.1, 0.1])
    c = Curator(cfg, keep).curate("a.png", Image.new("RGB", (10, 10)))
    assert c.verdict == "keep" and c.reason == "" and keep.json_calls == []

    drop = ScriptedVlm([0.1, 0.7, 0.2])
    c = Curator(cfg, drop).curate("b.png", Image.new("RGB", (10, 10)))
    assert c.verdict == "drop" and c.reason == "demasiado borrosa" and len(drop.json_calls) == 1
    assert c.probabilities == {"keep": 0.1, "drop": 0.7, "unsure": 0.2}


def test_prompt_uses_the_project_criterion_and_targets(tmp_path):
    cfg = make_cfg(tmp_path, 'curator: {keep_if: "se ve al menos un objeto de frente"}')
    vlm = ScriptedVlm([0.8, 0.1, 0.1])
    Curator(cfg, vlm).curate("a.png", Image.new("RGB", (10, 10)))
    assert "se ve al menos un objeto de frente" in vlm.prompt
    assert "a (cosa a), b (cosa b)" in vlm.prompt and "una escena" in vlm.prompt


def test_default_criterion_when_the_project_has_none(tmp_path):
    vlm = ScriptedVlm([0.8, 0.1, 0.1])
    Curator(make_cfg(tmp_path), vlm).curate("a.png", Image.new("RGB", (10, 10)))
    assert "clearly visible and in focus" in vlm.prompt


def test_save_and_load_roundtrip(tmp_path):
    cfg = make_cfg(tmp_path)
    Image.fromarray(np.zeros((20, 20, 3), np.uint8)).save(cfg.images_dir / "a.png")
    results = curate_images(cfg, ScriptedVlm([0.1, 0.8, 0.1]), ["a.png"])
    save_curation(tmp_path / "out" / "curation.json", results)
    assert load_curation(tmp_path / "out" / "curation.json") == {"a.png": results[0]}


def test_curated_keep_images_join_the_selection(tmp_path):
    cfg = make_cfg(tmp_path)
    records = [{"image": n, "verdict": v, "reasons": [], "metrics": None}
               for n, v in [("a.png", "keep"), ("b.png", "unsure"), ("c.png", "unsure"), ("d.png", "drop")]]
    (tmp_path / "out" / "quality.json").write_text(json.dumps({"images": records}), encoding="utf-8")

    assert [p.name for p in select_images(cfg)] == ["a.png"]                     # sin curador
    verdicts = {"b.png": "keep", "c.png": "drop"}
    (tmp_path / "out" / "curation.json").write_text(json.dumps({"images": [
        {"image": n, "verdict": v, "probabilities": {}, "reason": ""} for n, v in verdicts.items()
    ]}), encoding="utf-8")
    assert [p.name for p in select_images(cfg)] == ["a.png", "b.png"]            # b recuperada, c no
    assert [p.name for p in select_images(cfg, include_unsure=True)] == ["a.png", "b.png", "c.png"]
