"""Verificación por lotes de las cajas de ``labels.json`` y persistencia del resultado.

Por cada caja guarda lo que el VLM vio y una confianza combinada. Quién acepta o revisa cada
caja lo decide una etapa posterior con esos números.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from autolabel.calibration import Calibration
from autolabel.config import ProjectConfig
from autolabel.detectors import Detection
from autolabel.imaging import load_image
from autolabel.labeling import ImageLabels
from autolabel.verification import BoxVerdict, Verifier
from autolabel.vlm import VisionLanguageModel

log = logging.getLogger("autolabel.verifying")

_CHECKPOINT_EVERY = 10


@dataclass(frozen=True, slots=True)
class VerifiedBox:
    verdict: BoxVerdict
    confidence: float          # probabilidad calibrada de que la caja sea un objeto correcto

    @property
    def cls(self) -> str | None:
        return self.verdict.cls


@dataclass(frozen=True, slots=True)
class ImageVerification:
    image: str
    width: int
    height: int
    boxes: tuple[VerifiedBox, ...]


def fingerprint(cfg: ProjectConfig, calibration: Calibration | None, with_refs: bool) -> str:
    payload = [cfg.vlm.model, cfg.scene, cfg.detector.agnostic_prompt, with_refs,
               [(c.name, c.definition, c.not_this) for c in cfg.classes],
               None if calibration is None else calibration.signature()]
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _to_json(r: ImageVerification) -> dict:
    return {"image": r.image, "width": r.width, "height": r.height,
            "boxes": [{**asdict(b.verdict), "confidence": b.confidence} for b in r.boxes]}


def _from_json(d: dict) -> ImageVerification:
    boxes = []
    for b in d["boxes"]:
        confidence = b.pop("confidence")
        boxes.append(VerifiedBox(BoxVerdict(**{**b, "box": tuple(b["box"])}), confidence))
    return ImageVerification(d["image"], d["width"], d["height"], tuple(boxes))


def save_verification(path: Path, meta: dict, results: list[ImageVerification]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"meta": meta, "images": [_to_json(r) for r in results]}), encoding="utf-8")
    os.replace(tmp, path)


def load_verification(path: Path) -> tuple[dict, list[ImageVerification]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"], [_from_json(d) for d in data["images"]]


def verify_labels(
    cfg: ProjectConfig,
    vlm: VisionLanguageModel,
    labels: list[ImageLabels],
    *,
    calibration: Calibration | None,
    reference_sheet=None,
) -> list[ImageVerification]:
    """Verifica todas las cajas. Retoma lo ya hecho si la configuración no cambió."""
    out_file = cfg.out_dir / "verification.json"
    meta = {"fingerprint": fingerprint(cfg, calibration, reference_sheet is not None),
            "model": cfg.vlm.model, "calibrated": calibration is not None,
            "with_references": reference_sheet is not None}
    done: dict[str, ImageVerification] = {}
    if out_file.is_file():
        old_meta, old = load_verification(out_file)
        if old_meta.get("fingerprint") == meta["fingerprint"]:
            done = {r.image: r for r in old}

    verifier = Verifier(cfg, vlm, reference_sheet)
    results = {l.image: done[l.image] for l in labels if l.image in done}  # noqa: E741
    pending = [item for item in labels if item.image not in results]
    log.info("%d imágenes: %d ya verificadas, %d por procesar", len(labels), len(results), len(pending))

    start = time.perf_counter()
    for i, item in enumerate(pending, 1):
        img = load_image(cfg.images_dir / item.image)
        verdicts = verifier.verify_image(img, [Detection(d.cls, d.box, d.score) for d in item.detections])
        boxes = tuple(
            VerifiedBox(v, calibration.confidence(v) if calibration else v.det_score) for v in verdicts
        )
        results[item.image] = ImageVerification(item.image, item.width, item.height, boxes)
        if i % _CHECKPOINT_EVERY == 0 or i == len(pending):
            save_verification(out_file, meta, list(results.values()))
            log.info("  %d/%d  (%.1f s por imagen)", i, len(pending), (time.perf_counter() - start) / i)

    ordered = [results[item.image] for item in labels]
    save_verification(out_file, meta, ordered)
    return ordered
