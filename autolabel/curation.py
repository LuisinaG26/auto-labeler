"""Curador: decide con un modelo de visión-lenguaje si una imagen dudosa sirve para el dataset.

Actúa sobre lo que el filtro de calidad dejó como ``unsure``. El criterio de "sirve" sale de
``curator.keep_if`` en el YAML, y los objetivos de las clases del proyecto.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from autolabel import prompts
from autolabel.config import ProjectConfig
from autolabel.imaging import load_image
from autolabel.vlm import VisionLanguageModel

log = logging.getLogger("autolabel.curation")

Verdict = Literal["keep", "drop", "unsure"]
_VERDICTS: tuple[Verdict, ...] = ("keep", "drop", "unsure")
_DEFAULT_KEEP_IF = "at least one target object is clearly visible and in focus"
_REASON_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}},
    "required": ["reason"],
}


@dataclass(frozen=True, slots=True)
class Curation:
    image: str
    verdict: Verdict
    probabilities: dict[str, float]
    reason: str


class Curator:
    def __init__(self, cfg: ProjectConfig, vlm: VisionLanguageModel) -> None:
        self._vlm = vlm
        self._scene_line = f"Context: {cfg.scene}." if cfg.scene else ""
        self._targets = ", ".join(f"{c.name} ({c.prompt})" for c in cfg.classes)
        self._keep_if = cfg.curator.get("keep_if", _DEFAULT_KEEP_IF)
        self._prompt = prompts.render(
            "curator", scene_line=self._scene_line, targets=self._targets, keep_if=self._keep_if
        )
        self._options = [
            "keep: the photo is useful for the dataset",
            "drop: the photo is not useful",
            "unsure: it is impossible to tell",
        ]

    def curate(self, name: str, img: Image.Image) -> Curation:
        probs = self._vlm.choose([img], self._prompt, self._options)
        by_verdict = dict(zip(_VERDICTS, (probs[o] for o in self._options), strict=True))
        verdict = max(by_verdict, key=by_verdict.__getitem__)
        reason = ""
        if verdict != "keep":                       # el motivo solo importa cuando se aparta
            reason_prompt = prompts.render(
                "curator_reason", scene_line=self._scene_line, verdict=verdict,
                targets=self._targets, keep_if=self._keep_if,
            )
            reason = str(self._vlm.ask_json([img], reason_prompt, _REASON_SCHEMA).get("reason", "")).strip()
        return Curation(name, verdict, by_verdict, reason)


def curate_images(cfg: ProjectConfig, vlm: VisionLanguageModel, names: list[str]) -> list[Curation]:
    curator = Curator(cfg, vlm)
    results = []
    for i, name in enumerate(names, 1):
        img = load_image(cfg.images_dir / name, max_side=cfg.vlm.max_image_side)
        results.append(curator.curate(name, img))
        if i % 10 == 0 or i == len(names):
            log.info("  %d/%d", i, len(names))
    return results


def save_curation(path: Path, results: list[Curation]) -> None:
    data = {"images": [
        {"image": r.image, "verdict": r.verdict, "probabilities": r.probabilities, "reason": r.reason}
        for r in results
    ]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_curation(path: Path) -> dict[str, Curation]:
    data = json.loads(path.read_text(encoding="utf-8"))["images"]
    return {r["image"]: Curation(r["image"], r["verdict"], r["probabilities"], r["reason"]) for r in data}
