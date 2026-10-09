"""Verificación de cajas con un modelo de visión-lenguaje.

Para cada caja del detector se hacen dos preguntas sobre un recorte:
  1. ¿Hay exactamente un objeto, ninguno, o un grupo? Descarta falsos positivos y cajas que
     abarcan varios objetos juntos.
  2. ¿A qué clase del proyecto pertenece? Con las definiciones del YAML y, si existen,
     las imágenes de referencia.

Nada de esto conoce las clases: los textos salen de plantillas con variables y las
opciones se construyen con las clases del proyecto.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from autolabel import prompts
from autolabel.config import ProjectConfig
from autolabel.detectors import Detection
from autolabel.geometry import as_boxes
from autolabel.vlm import VisionLanguageModel

OBJECT_NONE, OBJECT_ONE, OBJECT_SEVERAL = "none", "one", "several"


@dataclass(frozen=True, slots=True)
class BoxVerdict:
    box: tuple[float, float, float, float]
    detector_class: str
    det_score: float
    p_none: float                      # probabilidad de que el recorte no tenga el objeto
    p_one: float                       # de que tenga exactamente uno
    p_several: float                   # de que sea un grupo
    cls: str | None                    # clase más probable según el VLM
    class_probs: dict[str, float]      # vacío si no se evaluó la clase
    group_geometry: bool               # la caja contiene casi enteras a otras dos o más


# ----------------------------------------------------------------------- recorte


def crop_candidate(
    img: Image.Image,
    box: Sequence[float],
    *,
    pad: float = 0.15,
    min_side: int = 224,
    max_side: int = 512,
) -> Image.Image:
    """Recorta la caja con un margen de contexto y la lleva a un tamaño que el VLM pueda leer."""
    x1, y1, x2, y2 = box
    px, py = max(pad * (x2 - x1), 4.0), max(pad * (y2 - y1), 4.0)
    region = (
        int(max(0, x1 - px)), int(max(0, y1 - py)),
        int(min(img.width, x2 + px)) + 1, int(min(img.height, y2 + py)) + 1,
    )
    crop = img.crop(region)
    scale = max(1.0, min_side / max(1, min(crop.size)))
    scale = min(scale, max_side / max(crop.size))
    if abs(scale - 1.0) > 1e-6:
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return crop


# ----------------------------------------------------------------------- geometría


def contains_others(
    boxes: Sequence[Sequence[float]],
    min_containment: float = 0.8,
    classes: Sequence[str] | None = None,
) -> np.ndarray:
    """Para cada caja, cuántas otras más chicas **de la misma clase** quedan casi enteras dentro.

    Una caja que contiene a dos o más del mismo tipo es sospechosa de abarcar un grupo. Es una
    señal barata y genérica, que se combina con la respuesta del VLM. Si se dan las clases, solo
    cuentan las de la misma clase: un objeto que contiene partes de otra clase (un camión y sus
    ruedas) no es un grupo.
    """
    b = as_boxes(boxes)
    if len(b) == 0:
        return np.zeros(0, dtype=int)
    area = np.prod(b[:, 2:] - b[:, :2], axis=1)
    top_left = np.maximum(b[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(b[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)       # inter[i, j]
    inside = inter / np.maximum(area[None, :], 1e-9)                         # fracción de j dentro de i
    contained = (inside >= min_containment) & (area[None, :] < area[:, None])
    if classes is not None:
        labels = np.asarray(classes)
        contained &= labels[:, None] == labels[None, :]
    return contained.sum(axis=1)


# ----------------------------------------------------------------------- textos


def _scene_line(cfg: ProjectConfig) -> str:
    return f"Context: {cfg.scene}." if cfg.scene else ""


def object_noun(cfg: ProjectConfig) -> str:
    """Cómo nombrar al objeto en general: el prompt agnóstico o, si no hay, las clases."""
    if cfg.detector.agnostic_prompt:
        return cfg.detector.agnostic_prompt
    return " or ".join(c.prompt for c in cfg.classes)


def _object_options(noun: str) -> list[str]:
    return [
        f"No {noun}: the center shows background or something else",
        f"Exactly one {noun}",
        f"Several {noun} together, a cluster or group",
    ]


def _class_lines(cfg: ProjectConfig) -> str:
    lines = []
    for c in cfg.classes:
        extra = f" (not: {', '.join(c.not_this)})" if c.not_this else ""
        lines.append(f"- {c.name}: {c.definition or c.prompt}{extra}")
    return "\n".join(lines)


# ----------------------------------------------------------------------- verificador


class Verifier:
    """Verifica las cajas de una imagen. Las referencias, si hay, se pasan como una hoja."""

    def __init__(
        self,
        cfg: ProjectConfig,
        vlm: VisionLanguageModel,
        reference_sheet: Image.Image | None = None,
        *,
        class_check_min_one: float = 0.2,
        crop_pad: float = 0.15,
    ) -> None:
        self._cfg = cfg
        self._vlm = vlm
        self._sheet = reference_sheet
        self._min_one = class_check_min_one
        self._pad = crop_pad
        self._noun = object_noun(cfg)
        self._object_prompt = prompts.render(
            "object_check", scene_line=_scene_line(cfg), noun=self._noun
        )
        self._object_options = _object_options(self._noun)
        reference_line = (
            "The first image is a reference sheet: each row shows examples of one class, labelled "
            "with its name. The second image is the crop to classify."
            if reference_sheet is not None else ""
        )
        self._class_prompt = prompts.render(
            "class_check", scene_line=_scene_line(cfg), reference_line=reference_line,
            noun=self._noun, class_definitions=_class_lines(cfg),
        )
        self._class_names = [c.name for c in cfg.classes]
        self._class_options = [f"{c.name}" for c in cfg.classes]

    def verify_box(self, img: Image.Image, det: Detection, *, group_geometry: bool = False) -> BoxVerdict:
        crop = crop_candidate(img, det.box, pad=self._pad)
        obj = self._vlm.choose([crop], self._object_prompt, self._object_options)
        p_none, p_one, p_several = (obj[o] for o in self._object_options)

        class_probs: dict[str, float] = {}
        cls: str | None = None
        if len(self._class_names) == 1:
            class_probs, cls = {self._class_names[0]: 1.0}, self._class_names[0]
        elif p_one >= self._min_one:
            images = [self._sheet, crop] if self._sheet is not None else [crop]
            probs = self._vlm.choose(images, self._class_prompt, self._class_options)
            class_probs = dict(probs)
            cls = max(class_probs, key=class_probs.__getitem__)

        return BoxVerdict(
            box=tuple(float(v) for v in det.box), detector_class=det.cls, det_score=det.score,
            p_none=p_none, p_one=p_one, p_several=p_several,
            cls=cls, class_probs=class_probs, group_geometry=group_geometry,
        )

    def verify_image(self, img: Image.Image, detections: Sequence[Detection]) -> list[BoxVerdict]:
        groups = contains_others([d.box for d in detections], classes=[d.cls for d in detections]) >= 2
        return [
            self.verify_box(img, d, group_geometry=bool(g))
            for d, g in zip(detections, groups, strict=True)
        ]
