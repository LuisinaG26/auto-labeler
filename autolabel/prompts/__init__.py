"""Plantillas de prompts con variables ``$nombre``.

Ninguna menciona clases concretas: todo lo específico del proyecto entra por las variables,
que se rellenan desde ``project.yaml``.
"""
from __future__ import annotations

from importlib import resources
from string import Template


def render(name: str, **variables: str) -> str:
    """Rellena la plantilla ``<name>.txt``. Las variables sin valor quedan vacías."""
    text = (resources.files(__package__) / f"{name}.txt").read_text(encoding="utf-8")
    values = {k: v for k, v in variables.items()}
    rendered = Template(text).safe_substitute(values)
    # Las líneas que quedaron vacías por una variable opcional no deben dejar huecos
    return "\n".join(line for line in rendered.splitlines() if line.strip() or line == "").strip()
