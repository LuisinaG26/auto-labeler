"""Esquema y validación de ``project.yaml``.

Todo lo que cambia entre proyectos vive en ese archivo. El código no debe
conocer nombres de clases concretos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ClassConfig",
    "DetectorConfig",
    "LabelingConfig",
    "ProjectConfig",
    "QualityConfig",
    "References",
    "StrategyConfig",
    "Thresholds",
    "VlmConfig",
    "load_project",
]

class _Strict(BaseModel):
    """Base: rechaza claves desconocidas, así un error de tipeo en el YAML falla al cargar."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(_Strict):
    auto_accept: float = Field(0.85, ge=0.0, le=1.0)
    review: float = Field(0.40, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.review > self.auto_accept:
            raise ValueError("thresholds.review no puede superar a thresholds.auto_accept")
        return self


class References(_Strict):
    positive: tuple[str, ...] = ()
    negative: tuple[str, ...] = ()


class ClassConfig(_Strict):
    name: str = Field(min_length=1)
    prompt: str = Field(min_length=1)                 # texto que recibe el detector
    synonyms: tuple[str, ...] = ()
    definition: str = ""
    not_this: tuple[str, ...] = ()
    source_labels: tuple[str, ...] = ()               # etiquetas del dataset de origen
    references: References = References()
    min_box_px: int = Field(0, ge=0)
    thresholds: Thresholds = Thresholds()


class StrategyConfig(_Strict):
    """Una forma de correr el detector: modelo, umbrales y preproceso."""

    backend: str = "grounding_dino"
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = Field(0.25, ge=0.0, le=1.0)
    text_threshold: float = Field(0.25, ge=0.0, le=1.0)
    nms_iou: float = Field(0.5, gt=0.0, le=1.0)
    tiles: int = Field(1, ge=1, le=4)                  # partir la imagen en tiles x tiles: objetos chicos


class DetectorConfig(StrategyConfig):
    """Estrategia por defecto del proyecto, más el prompt agnóstico."""

    # Prompt que agrupa todas las clases en una, para medir localización sin clasificar
    agnostic_prompt: str | None = None


class LabelingConfig(_Strict):
    """Bucle etiquetado y verificación: estrategias a probar en orden y cuándo parar."""

    mode: Literal["agnostic", "direct"] = "agnostic"
    strategies: tuple[StrategyConfig, ...] = ()        # vacío: una sola, la del bloque `detector`
    max_iterations: int = Field(3, ge=1)
    min_gain: float = Field(0.02, ge=0.0, le=1.0)      # mejora mínima de lo resuelto para seguir probando
    time_budget_min: float | None = Field(None, gt=0)


class QualityConfig(_Strict):
    """Cortes del filtro de calidad. Se calibran por proyecto."""

    normalize_side: int = Field(500, ge=64)            # lado mayor al que se mide el desenfoque
    min_resolution: int = Field(300, ge=1)             # lado menor mínimo, en píxeles
    # Cortes de nitidez (ver autolabel.quality.sharpness). Su escala depende de la cámara y no
    # hay valor universal: sin ellos no se filtra por desenfoque. Se obtienen con
    # `autolabel quality --calibrate`, que los sugiere y deja una hoja de muestras para confirmarlos.
    blur_min: float | None = Field(None, ge=0.0)       # por debajo: se aparta
    blur_ok: float | None = Field(None, ge=0.0)        # por encima: pasa. Entre ambos: dudosa
    brightness_range: tuple[float, float] = (25.0, 235.0)   # brillo medio aceptado, de 0 a 255
    clipped_max: float = Field(0.35, ge=0.0, le=1.0)   # fracción máxima de píxeles negros o quemados
    dedupe: bool = True
    dedupe_max_distance: int = Field(4, ge=0, le=64)   # distancia de Hamming entre hashes

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.blur_min is None) != (self.blur_ok is None):
            raise ValueError("quality.blur_min y quality.blur_ok van juntos: definir ambos o ninguno")
        if self.blur_min is not None and self.blur_min > self.blur_ok:
            raise ValueError("quality.blur_min no puede superar a quality.blur_ok")
        if self.brightness_range[0] >= self.brightness_range[1]:
            raise ValueError("quality.brightness_range debe ser [mínimo, máximo]")
        return self


class VlmConfig(_Strict):
    """Modelo de visión-lenguaje para curar y verificar. Detrás de una interfaz común."""

    provider: Literal["ollama"] = "ollama"
    # Variante de instrucción: la de razonamiento emite un bloque de pensamiento antes de responder
    model: str = "qwen3-vl:4b-instruct"
    base_url: str = "http://ollama:11434"
    max_image_side: int = Field(1024, ge=64)


class ProjectConfig(_Strict):
    name: str = Field(min_length=1)
    root: Path                                        # carpeta del proyecto, la fija load_project
    images: str = "./images"
    scene: str = ""
    classes: tuple[ClassConfig, ...] = Field(min_length=1)
    detector: DetectorConfig = DetectorConfig()
    quality: QualityConfig = QualityConfig()
    vlm: VlmConfig = VlmConfig()
    labeling: LabelingConfig = LabelingConfig()
    curator: dict[str, str] = Field(default_factory=dict)
    export: tuple[Literal["yolo", "coco"], ...] = ("yolo",)

    @model_validator(mode="after")
    def _unique_names_and_labels(self) -> Self:
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError(f"Nombres de clase repetidos: {names}")
        seen: dict[str, str] = {}
        for c in self.classes:
            for label in c.source_labels:
                if label in seen:
                    raise ValueError(
                        f"La etiqueta de origen {label!r} está en {seen[label]!r} y en {c.name!r}"
                    )
                seen[label] = c.name
        return self

    @property
    def images_dir(self) -> Path:
        return (self.root / self.images).resolve()

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.classes)

    def class_by_name(self, name: str) -> ClassConfig:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(f"El proyecto no tiene la clase {name!r}")

    def strategies(self) -> list[StrategyConfig]:
        """Estrategias del bucle, ya recortadas al máximo de iteraciones."""
        chosen = list(self.labeling.strategies) or [StrategyConfig(**self.detector.model_dump(
            exclude={"agnostic_prompt"}))]
        return chosen[: self.labeling.max_iterations]

    def label_map(self) -> dict[str, str]:
        """Etiqueta del dataset de origen -> nombre de clase del proyecto."""
        return {label: c.name for c in self.classes for label in c.source_labels}


def load_project(path: str | Path) -> ProjectConfig:
    """Carga un proyecto desde su carpeta o desde el ``project.yaml``."""
    p = Path(path)
    if p.is_dir():
        p = p / "project.yaml"
    if not p.is_file():
        raise FileNotFoundError(f"No existe {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if "root" in data:
        raise ValueError("'root' lo calcula el cargador y no debe estar en project.yaml")
    return ProjectConfig(root=p.parent.resolve(), **data)
