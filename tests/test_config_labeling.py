import textwrap

import pytest
from pydantic import ValidationError

from autolabel.config import load_project

BASE = """
name: demo
classes:
  - {name: a, prompt: "pa"}
detector:
  model_id: modelo-base
  box_threshold: 0.3
  agnostic_prompt: "cosa"
"""


def load(tmp_path, extra=""):
    (tmp_path / "project.yaml").write_text(textwrap.dedent(BASE) + textwrap.dedent(extra), encoding="utf-8")
    return load_project(tmp_path)


def test_without_a_labeling_block_there_is_one_strategy_from_the_detector(tmp_path):
    strategies = load(tmp_path).strategies()
    assert len(strategies) == 1
    assert strategies[0].model_id == "modelo-base" and strategies[0].box_threshold == 0.3
    assert strategies[0].tiles == 1


def test_strategies_are_trimmed_to_max_iterations(tmp_path):
    cfg = load(tmp_path, """
        labeling:
          max_iterations: 2
          strategies:
            - {model_id: m1}
            - {model_id: m2, tiles: 2}
            - {model_id: m3}
    """)
    assert [s.model_id for s in cfg.strategies()] == ["m1", "m2"]
    assert cfg.strategies()[1].tiles == 2


def test_labeling_block_rejects_unknown_keys_and_bad_values(tmp_path):
    with pytest.raises(ValidationError):
        load(tmp_path, "labeling: {max_iteraciones: 3}\n")
    with pytest.raises(ValidationError):
        load(tmp_path, "labeling: {strategies: [{tiles: 9}]}\n")
    with pytest.raises(ValidationError):
        load(tmp_path, "labeling: {mode: otro}\n")
