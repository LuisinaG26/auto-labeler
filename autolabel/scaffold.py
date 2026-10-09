"""Andamiaje de un proyecto nuevo: carpeta y ``project.yaml`` de plantilla.

La plantilla trae solo lo imprescindible y comenta lo que hay que completar o calibrar, así
empezar con otro dominio no exige copiar y editar el archivo de otro proyecto.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

_NAME = re.compile(r"[^A-Za-z0-9_\-]+")


def _quote(text: str) -> str:
    """Una cadena YAML segura: JSON es un subconjunto de YAML."""
    return json.dumps(text, ensure_ascii=False)


def project_template(name: str, classes: Sequence[str], images: str, scene: str) -> str:
    if not classes:
        raise ValueError("Hace falta al menos una clase")
    if len(set(classes)) != len(classes):
        raise ValueError(f"Clases repetidas: {list(classes)}")
    class_blocks = "\n".join(
        f"""  - name: {_quote(c)}
    prompt: {_quote(c)}              # texto que entiende el detector: mejor en inglés y descriptivo
    definition: ""                  # cómo reconocer la clase, para el modelo de visión
    not_this: []                    # cosas parecidas que NO son esta clase
    min_box_px: 10                  # lado menor mínimo de una caja, en píxeles
    thresholds: {{ auto_accept: 0.85, review: 0.40 }}   # calibrar con el set dorado"""
        for c in classes
    )
    return f"""name: {_quote(name)}
images: {_quote(images)}
scene: {_quote(scene)}     # contexto para el curador y el verificador

classes:
{class_blocks}

detector:
  backend: grounding_dino
  model_id: IDEA-Research/grounding-dino-tiny
  box_threshold: 0.25
  agnostic_prompt: ""               # un prompt que agrupe todas las clases (por ejemplo el objeto en general)

# Cortes de nitidez: sin ellos no se filtra por desenfoque. Obtenerlos con
#   python -m autolabel.cli quality --project <carpeta> --calibrate
# quality:
#   blur_min: 0
#   blur_ok: 0

curator:
  keep_if: "at least one target object is clearly visible and in focus"

labeling:
  mode: agnostic                    # agnostic: un prompt para todo y la clase la decide el verificador
  max_iterations: 1                 # más rondas solo valen la pena con estrategias que vean cosas distintas
  min_gain: 0.02
  strategies:                       # se prueban en orden sobre lo que quede dudoso
    - {{ model_id: IDEA-Research/grounding-dino-tiny, box_threshold: 0.2 }}
    # Un modelo más grande o un umbral más bajo del mismo modelo suelen compartir los puntos ciegos
    # del primero y casi no suman. Medirlo con eval_pipeline antes de agregarlo, y subir max_iterations.

export: [yolo]
"""


def init_project(
    folder: str | Path,
    classes: Sequence[str],
    *,
    images: str = "./images",
    scene: str = "",
    force: bool = False,
) -> Path:
    """Crea la carpeta del proyecto con su ``project.yaml``. Devuelve la ruta del archivo."""
    folder = Path(folder)
    target = folder / "project.yaml"
    if target.exists() and not force:
        raise FileExistsError(f"{target} ya existe: usa --force para reemplazarlo")
    name = _NAME.sub("-", folder.resolve().name).strip("-") or "proyecto"
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(project_template(name, classes, images, scene), encoding="utf-8")
    if images == "./images":
        (folder / "images").mkdir(exist_ok=True)
    return target
