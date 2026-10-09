"""Lógica del bucle etiquetado y verificación, sin LangGraph ni modelos.

Cada imagen lleva su propio estado: las cajas que se fueron encontrando, qué estrategia las
produjo, lo que dijo la verificación y la decisión actual. Las funciones de este módulo son
puras o solo modifican el estado que reciben, así se prueban sin GPU.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from PIL import Image

from autolabel.calibration import Calibration
from autolabel.config import ProjectConfig
from autolabel.detectors import Detection
from autolabel.geometry import iou_matrix
from autolabel.labeling import ImageLabels
from autolabel.verification import BoxVerdict, Verifier, contains_others

BoxStatus = Literal["pending", "accepted", "uncertain", "rejected"]
ImageStatus = Literal["active", "accepted", "review"]
_MERGE_IOU = 0.5


@dataclass(slots=True)
class BoxState:
    box: tuple[float, float, float, float]
    det_score: float
    detector_class: str
    strategies: list[int] = field(default_factory=list)    # estrategias que encontraron esta caja
    models: list[str] = field(default_factory=list)        # modelos distintos que la encontraron
    verdict: BoxVerdict | None = None
    confidence: float | None = None
    status: BoxStatus = "pending"

    @property
    def cls(self) -> str | None:
        return self.verdict.cls if self.verdict else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = list(self.box)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BoxState:
        v = d["verdict"]
        verdict = BoxVerdict(**{**v, "box": tuple(v["box"])}) if v else None
        return cls(tuple(d["box"]), d["det_score"], d["detector_class"], list(d["strategies"]),
                   list(d["models"]), verdict, d["confidence"], d["status"])


@dataclass(slots=True)
class ImageState:
    name: str
    width: int = 0
    height: int = 0
    boxes: list[BoxState] = field(default_factory=list)
    status: ImageStatus = "active"
    rounds: int = 0                                         # cuántas rondas de etiquetado pasó
    resolved: list[float] = field(default_factory=list)     # fracción de cajas resueltas tras cada ronda
    reason: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "width": self.width, "height": self.height,
                "boxes": [b.to_dict() for b in self.boxes], "status": self.status,
                "rounds": self.rounds, "resolved": list(self.resolved), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> ImageState:
        return cls(d["name"], d["width"], d["height"], [BoxState.from_dict(b) for b in d["boxes"]],
                   d["status"], d["rounds"], list(d["resolved"]), d["reason"])


# ----------------------------------------------------------------------- fusión


def merge_detections(
    image: ImageState, dets: Sequence[Detection], strategy: int, model_id: str, iou: float = _MERGE_IOU
) -> int:
    """Suma las detecciones de una estrategia a las cajas que ya tiene la imagen.

    Una detección que coincide con una caja existente no crea otra: registra que la
    estrategia (y su modelo) también la encontró. Devuelve cuántas cajas nuevas se crearon.
    """
    new = 0
    existing = [b.box for b in image.boxes]
    for d in dets:
        match = None
        if existing:
            ious = iou_matrix([d.box], existing)[0]
            best = int(ious.argmax())
            if ious[best] >= iou:
                match = image.boxes[best]
        if match is not None:
            if strategy not in match.strategies:
                match.strategies.append(strategy)
            if model_id not in match.models:
                match.models.append(model_id)
        else:
            image.boxes.append(BoxState(tuple(d.box), d.score, d.cls, [strategy], [model_id]))
            existing.append(tuple(d.box))
            new += 1
    return new


# ----------------------------------------------------------------------- verificación


def verify_pending(
    image: ImageState, img: Image.Image, verifier: Verifier, calibration: Calibration | None
) -> int:
    """Verifica las cajas nuevas y recalcula la confianza de todas.

    La geometría de grupo depende del conjunto de cajas, así que se refresca en todas: una caja
    vieja puede pasar a contener a las nuevas. Las respuestas del VLM no se repiten.
    """
    groups = contains_others(
        [b.box for b in image.boxes], classes=[b.detector_class for b in image.boxes]
    ) >= 2
    done = 0
    for box, is_group in zip(image.boxes, groups, strict=True):
        if box.verdict is None:
            det = Detection(box.detector_class, box.box, box.det_score)
            box.verdict = verifier.verify_box(img, det, group_geometry=bool(is_group))
            done += 1
        elif box.verdict.group_geometry != bool(is_group):
            box.verdict = replace(box.verdict, group_geometry=bool(is_group))
        box.confidence = calibration.confidence(box.verdict) if calibration else box.verdict.det_score
    return done


# ----------------------------------------------------------------------- decisión


def classify_box(box: BoxState, cfg: ProjectConfig) -> BoxStatus:
    """Aceptar, dudar o rechazar una caja según los umbrales de su clase."""
    if box.verdict is None or box.confidence is None:
        return "pending"
    if box.cls is None:                          # el VLM no vio un objeto
        return "rejected"
    thresholds = cfg.class_by_name(box.cls).thresholds
    if box.confidence >= thresholds.auto_accept:
        return "accepted"
    if box.confidence >= thresholds.review:
        return "uncertain"
    return "rejected"


def decide_image(
    image: ImageState, cfg: ProjectConfig, *, can_retry: bool, min_gain: float
) -> ImageStatus:
    """Actualiza el estado de las cajas y decide qué sigue con la imagen.

    - Sin cajas dudosas: aceptada.
    - Con dudosas y margen para reintentar: sigue activa, salvo que el último intento no haya
      mejorado lo resuelto en ``min_gain``.
    - Si no: pasa a revisión humana, con el motivo.
    """
    for b in image.boxes:
        b.status = classify_box(b, cfg)
    total = len(image.boxes)
    uncertain = sum(b.status == "uncertain" for b in image.boxes)
    image.resolved.append((total - uncertain) / total if total else 0.0)
    image.rounds += 1

    if total and not uncertain:
        image.status, image.reason = "accepted", ""
    elif not can_retry:
        image.status = "review"
        image.reason = "sin cajas" if not total else f"cajas dudosas: {uncertain}"
    elif total and len(image.resolved) >= 2 and image.resolved[-1] - image.resolved[-2] < min_gain:
        # Con cajas se mide la mejora. Sin ninguna no hay qué medir: se sigue hasta agotar estrategias
        image.status, image.reason = "review", f"sin mejora tras {image.rounds} intentos"
    else:
        image.status = "active"
    return image.status


# ----------------------------------------------------------------------- salidas


def accepted_labels(image: ImageState) -> ImageLabels:
    dets = tuple(Detection(b.cls, b.box, b.confidence) for b in image.boxes if b.status == "accepted")
    return ImageLabels(image.name, image.width, image.height, dets)


def review_labels(image: ImageState) -> ImageLabels:
    """Cajas para que el humano corrija: las aceptadas y las dudosas, con su clase propuesta."""
    keep = ("accepted", "uncertain")
    dets = tuple(Detection(b.cls, b.box, b.confidence) for b in image.boxes if b.status in keep)
    return ImageLabels(image.name, image.width, image.height, dets)
