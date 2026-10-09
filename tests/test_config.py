import textwrap

import pytest
from pydantic import ValidationError

from autolabel.config import load_project

VALID = """
name: demo
classes:
  - name: a
    prompt: "thing a"
    source_labels: [x, y]
  - name: b
    prompt: "thing b"
    source_labels: [z]
"""


def write(tmp_path, text):
    (tmp_path / "project.yaml").write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def test_loads_from_folder_and_resolves_paths(tmp_path):
    cfg = load_project(write(tmp_path, VALID))
    assert cfg.class_names == ("a", "b")
    assert cfg.images_dir == (tmp_path / "images").resolve()
    assert cfg.label_map() == {"x": "a", "y": "a", "z": "b"}
    assert cfg.detector.backend == "grounding_dino"


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ValidationError):
        load_project(write(tmp_path, VALID + "clases_extra: 1\n"))


def test_duplicate_class_names_are_rejected(tmp_path):
    text = """
    name: demo
    classes:
      - {name: a, prompt: p}
      - {name: a, prompt: q}
    """
    with pytest.raises(ValidationError, match="repetidos"):
        load_project(write(tmp_path, text))


def test_source_label_in_two_classes_is_rejected(tmp_path):
    text = """
    name: demo
    classes:
      - {name: a, prompt: p, source_labels: [x]}
      - {name: b, prompt: q, source_labels: [x]}
    """
    with pytest.raises(ValidationError, match="origen"):
        load_project(write(tmp_path, text))


def test_thresholds_must_be_ordered(tmp_path):
    text = """
    name: demo
    classes:
      - name: a
        prompt: p
        thresholds: {auto_accept: 0.3, review: 0.6}
    """
    with pytest.raises(ValidationError):
        load_project(write(tmp_path, text))


def test_requires_at_least_one_class(tmp_path):
    with pytest.raises(ValidationError):
        load_project(write(tmp_path, "name: demo\nclasses: []\n"))


def test_missing_project_and_reserved_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nada")
    with pytest.raises(ValueError, match="root"):
        load_project(write(tmp_path, VALID + "root: /tmp\n"))
