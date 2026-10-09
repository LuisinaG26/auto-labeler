"""Etiquetado por lotes: elige las imágenes que pasaron el filtro y corre el detector.

El resultado se guarda con los scores originales para que las etapas siguientes
(verificación, revisión) puedan decidir con ellos. Si se interrumpe, al repetir el
comando se retoma desde donde quedó siempre que la configuración sea la misma.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from autolabel.config import ProjectConfig
from autolabel.detectors import Detection, Query
from autolabel.geometry import box_sizes

log = logging.getLogger("autolabel.labeling")

Mode = Literal["agnostic", "direct"]
_CHECKPOINT_EVERY = 25


@dataclass(frozen=True, slots=True)
class ImageLabels:
    image: str                         # nombre de archivo dentro de la carpeta de imágenes
    width: int                         # tamaño con la orientación EXIF aplicada
    height: int
    detections: tuple[Detection, ...]


class AsyncDetector(Protocol):
    async def detect(self, path: Path): ...    # devuelve autolabel.detector_client.DetectOutput


# ----------------------------------------------------------------------- selección


def _curated_keep(cfg: ProjectConfig) -> set[str]:
    """Imágenes dudosas que el curador resolvió como ``keep``. Vacío si aún no corrió."""
    from autolabel.curation import load_curation

    path = cfg.out_dir / "curation.json"
    if not path.is_file():
        return set()
    return {name for name, c in load_curation(path).items() if c.verdict == "keep"}


def select_images(
    cfg: ProjectConfig, *, include_unsure: bool = False, limit: int | None = None, seed: int = 0
) -> list[Path]:
    """Imágenes que pasaron el filtro de calidad, opcionalmente una muestra reproducible.

    Las dudosas entran si el curador las dejó en ``keep``, o todas con ``include_unsure``.
    """
    quality_file = cfg.out_dir / "quality.json"
    if not quality_file.is_file():
        raise FileNotFoundError(
            f"Falta {quality_file}. El filtro de calidad corre antes que el etiquetado: "
            f"python -m autolabel.cli quality --project {cfg.root}"
        )
    records = json.loads(quality_file.read_text(encoding="utf-8"))["images"]
    curated = _curated_keep(cfg)
    names = sorted(
        r["image"] for r in records
        if r["verdict"] == "keep"
        or (r["verdict"] == "unsure" and (include_unsure or r["image"] in curated))
    )
    if limit is not None and limit < len(names):
        names = sorted(random.Random(seed).sample(names, limit))
    return [cfg.images_dir / n for n in names]


def build_queries(cfg: ProjectConfig, mode: Mode) -> list[Query]:
    """Consultas al detector. En modo agnóstico se busca todo con un solo prompt."""
    if mode == "direct":
        return [Query(c.name, c.prompt) for c in cfg.classes]
    prompt = cfg.detector.agnostic_prompt
    if not prompt:
        raise ValueError("El modo 'agnostic' necesita detector.agnostic_prompt en project.yaml")
    return [Query(prompt, prompt)]


def _min_box_px(cfg: ProjectConfig, class_name: str) -> int:
    """Tamaño mínimo de caja para una clase. Para el prompt agnóstico, el menor del proyecto."""
    for c in cfg.classes:
        if c.name == class_name:
            return c.min_box_px
    return min(c.min_box_px for c in cfg.classes)


def drop_tiny_boxes(cfg: ProjectConfig, dets: tuple[Detection, ...]) -> tuple[Detection, ...]:
    if not dets:
        return dets
    sizes = box_sizes([d.box for d in dets]).min(axis=1)
    return tuple(d for d, s in zip(dets, sizes, strict=True) if s >= _min_box_px(cfg, d.cls))


# ----------------------------------------------------------------------- persistencia


def fingerprint(cfg: ProjectConfig, queries: list[Query]) -> str:
    """Identifica la configuración que produjo las detecciones, para saber si se pueden reutilizar."""
    payload = {
        "detector": cfg.detector.model_dump(),
        "queries": [(q.name, q.prompt) for q in queries],
        "min_box_px": {c.name: c.min_box_px for c in cfg.classes},
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def save_labels(path: Path, meta: dict, results: list[ImageLabels]) -> None:
    data = {
        "meta": meta,
        "images": [
            {
                "image": r.image, "width": r.width, "height": r.height,
                "detections": [{"cls": d.cls, "box": list(d.box), "score": d.score} for d in r.detections],
            }
            for r in results
        ],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)                      # escritura atómica: un corte no deja un archivo roto


def load_labels(path: Path) -> tuple[dict, list[ImageLabels]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = [
        ImageLabels(
            r["image"], r["width"], r["height"],
            tuple(Detection(d["cls"], tuple(d["box"]), d["score"]) for d in r["detections"]),
        )
        for r in data["images"]
    ]
    return data["meta"], results


# ----------------------------------------------------------------------- etiquetado


async def label_images(
    cfg: ProjectConfig, detector: AsyncDetector, paths: list[Path], queries: list[Query], mode: Mode
) -> list[ImageLabels]:
    """Etiqueta las imágenes dadas, retomando lo ya hecho con la misma configuración."""
    out_file = cfg.out_dir / "labels.json"
    meta = {"fingerprint": fingerprint(cfg, queries), "mode": mode,
            "classes": [q.name for q in queries], "model_id": cfg.detector.model_id,
            "box_threshold": cfg.detector.box_threshold}

    done: dict[str, ImageLabels] = {}
    if out_file.is_file():
        old_meta, old = load_labels(out_file)
        if old_meta.get("fingerprint") == meta["fingerprint"]:
            done = {r.image: r for r in old}
        else:
            log.info("Configuración distinta a la de la corrida anterior: se etiqueta desde cero")

    wanted = {p.name for p in paths}
    results = {name: r for name, r in done.items() if name in wanted}
    pending = [p for p in paths if p.name not in results]
    log.info("%d imágenes: %d ya etiquetadas, %d por procesar", len(paths), len(results), len(pending))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for i, path in enumerate(pending, 1):
        out = await detector.detect(path)
        kept = drop_tiny_boxes(cfg, out.detections)
        results[path.name] = ImageLabels(path.name, out.width, out.height, kept)
        if i % _CHECKPOINT_EVERY == 0 or i == len(pending):
            save_labels(out_file, meta, list(results.values()))
            rate = i / (time.perf_counter() - start)
            log.info("  %d/%d  (%.1f imágenes/s)", i, len(pending), rate)

    ordered = [results[p.name] for p in paths]
    save_labels(out_file, meta, ordered)
    return ordered
