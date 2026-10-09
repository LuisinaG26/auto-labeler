"""Filtro de calidad de imágenes: aparta lo inservible antes de gastar un detector.

Mide desenfoque, exposición, resolución y duplicados. No usa modelos. El
resultado por imagen es ``keep``, ``drop`` o ``unsure`` con el motivo, para que
nada se descarte sin explicación ni se borre del dataset original.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from autolabel.config import QualityConfig
from autolabel.imaging import image_size, load_image

Verdict = Literal["keep", "drop", "unsure"]

_CLIP_LOW, _CLIP_HIGH = 5, 250       # píxeles casi negros o casi blancos
_BLUR_GRID = 4                       # la imagen se divide en una grilla de 4x4 zonas
_BLUR_PERCENTILE = 90                # y se toma el percentil 90 de la nitidez por zona


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    width: int                      # tamaño original, con orientación aplicada
    height: int
    blur: float                     # nitidez a escala normalizada, ver sharpness(): más alto, más nítida
    brightness: float               # brillo medio, de 0 a 255
    clipped: float                  # fracción de píxeles negros o quemados
    phash: int                      # hash perceptual de 64 bits


@dataclass(frozen=True, slots=True)
class Assessment:
    image: str
    verdict: Verdict
    reasons: tuple[str, ...]
    metrics: QualityMetrics | None  # None si la imagen no se pudo leer


# ----------------------------------------------------------------------- medidas


def sharpness(gray: np.ndarray) -> float:
    """Nitidez como percentil alto de la varianza del Laplaciano por zonas.

    La varianza global castiga las escenas con superficies lisas (una hoja grande y
    nítida) y premia las muy texturadas. Medir por zonas y tomar un percentil alto
    pregunta si hay bordes bien definidos en alguna parte, que es lo que se pierde
    con el desenfoque. En imágenes reales de campo coincide mejor con el criterio visual.
    """
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    zh, zw = lap.shape[0] // _BLUR_GRID, lap.shape[1] // _BLUR_GRID
    # Se recortan los píxeles sobrantes del borde y se agrupa en zonas sin copiar datos
    zones = lap[: zh * _BLUR_GRID, : zw * _BLUR_GRID].reshape(_BLUR_GRID, zh, _BLUR_GRID, zw)
    return float(np.percentile(zones.var(axis=(1, 3)), _BLUR_PERCENTILE))


def _perceptual_hash(gray: np.ndarray) -> int:
    """Hash perceptual de 64 bits: las 8x8 frecuencias bajas de la DCT, comparadas con su mediana."""
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = low > np.median(low[1:])          # se excluye la componente continua
    return int(np.packbits(bits).view(">u8")[0])


def measure_image(path: str | Path, normalize_side: int = 500) -> QualityMetrics:
    """Mide una imagen. El desenfoque se calcula a una escala fija para que sea comparable."""
    width, height = image_size(path)
    gray = cv2.cvtColor(np.asarray(load_image(path, max_side=normalize_side)), cv2.COLOR_RGB2GRAY)
    return QualityMetrics(
        width=width,
        height=height,
        blur=sharpness(gray),
        brightness=float(gray.mean()),
        clipped=float(((gray <= _CLIP_LOW) | (gray >= _CLIP_HIGH)).mean()),
        phash=_perceptual_hash(gray),
    )


# ----------------------------------------------------------------------- duplicados


def find_duplicate_pairs(hashes: Sequence[int], max_distance: int) -> list[tuple[int, int]]:
    """Pares de índices con distancia de Hamming <= max_distance.

    Vectorizado por fila: O(n^2) comparaciones pero sin bucle en Python por par.
    """
    arr = np.asarray(hashes, dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    for i in range(len(arr) - 1):
        distance = np.bitwise_count(arr[i + 1 :] ^ arr[i])
        pairs.extend((i, i + 1 + int(j)) for j in np.flatnonzero(distance <= max_distance))
    return pairs


def _groups(n: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Componentes conexas de los pares (unión-búsqueda), solo las de más de un elemento."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)
    members: dict[int, list[int]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    return [g for g in members.values() if len(g) > 1]


# ----------------------------------------------------------------------- veredicto


def judge(m: QualityMetrics, cfg: QualityConfig) -> tuple[Verdict, tuple[str, ...]]:
    """Veredicto de una imagen por sí sola, sin considerar duplicados."""
    reasons: list[str] = []
    if min(m.width, m.height) < cfg.min_resolution:
        reasons.append("low_resolution")
    low, high = cfg.brightness_range
    if m.brightness < low:
        reasons.append("too_dark")
    elif m.brightness > high:
        reasons.append("too_bright")
    if m.clipped > cfg.clipped_max:
        reasons.append("clipped")
    if reasons:
        return "drop", tuple(reasons)
    if cfg.blur_min is None or cfg.blur_ok is None:     # sin cortes de nitidez no se filtra por eso
        return "keep", ()
    if m.blur < cfg.blur_min:
        return "drop", ("blurry",)
    if m.blur < cfg.blur_ok:
        return "unsure", ("slightly_blurry",)
    return "keep", ()


def measure_many(
    paths: Sequence[str | Path], normalize_side: int, workers: int | None = None
) -> list[QualityMetrics | None]:
    """Mide un lote en paralelo. Una imagen ilegible da ``None`` y no frena al resto."""

    def measure(path: str | Path) -> QualityMetrics | None:
        try:
            return measure_image(path, normalize_side)
        except (OSError, ValueError):       # archivo corrupto o formato no soportado
            return None

    # cv2 y PIL liberan el GIL en las partes pesadas, así que los hilos escalan
    with ThreadPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 1)) as pool:
        return list(pool.map(measure, paths))


def assess_batch(
    paths: Sequence[str | Path], cfg: QualityConfig, workers: int | None = None
) -> list[Assessment]:
    """Evalúa un lote en paralelo y marca los duplicados conservando la imagen más nítida."""
    paths = [str(p) for p in paths]
    measured = measure_many(paths, cfg.normalize_side, workers)

    verdicts: list[tuple[Verdict, tuple[str, ...]]] = [
        ("drop", ("unreadable",)) if m is None else judge(m, cfg) for m in measured
    ]

    if cfg.dedupe:
        ok = [i for i, m in enumerate(measured) if m is not None]
        hashes = [measured[i].phash for i in ok]            # type: ignore[union-attr]
        pairs = [(ok[a], ok[b]) for a, b in find_duplicate_pairs(hashes, cfg.dedupe_max_distance)]
        for group in _groups(len(paths), pairs):
            candidates = [i for i in group if verdicts[i][0] != "drop"]
            if not candidates:
                continue
            best = max(candidates, key=lambda i: measured[i].blur)   # type: ignore[union-attr]
            for i in candidates:
                if i != best:
                    verdicts[i] = ("drop", (f"duplicate_of:{Path(paths[best]).name}",))

    return [
        Assessment(path, verdict, reasons, m)
        for path, m, (verdict, reasons) in zip(paths, measured, verdicts, strict=True)
    ]


def assessment_record(a: Assessment) -> dict:
    """Forma de una evaluación en ``quality.json``. El hash de 64 bits va como texto."""
    metrics = asdict(a.metrics) if a.metrics else None
    if metrics:
        metrics["phash"] = f"{metrics['phash']:016x}"
    return {"image": Path(a.image).name, "verdict": a.verdict, "reasons": list(a.reasons), "metrics": metrics}


def save_assessments(path: Path, params: dict, results: Sequence[Assessment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"params": params, "images": [assessment_record(a) for a in results]}, indent=1),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------- calibración de cortes

_ANCHOR_SIGMAS = (1.0, 2.0, 3.0)      # desenfoques gaussianos de referencia, a la escala normalizada
_REFERENCE_FRACTION = 0.2             # se toman las más nítidas del conjunto como imágenes de referencia
_PERCENTILES = (5, 10, 20, 30, 50, 75, 90)


def calibration_report(
    paths: Sequence[str | Path], normalize_side: int = 500, workers: int | None = None
) -> dict:
    """Datos para fijar los cortes de nitidez de un proyecto.

    Además de los percentiles, que son relativos al conjunto, calcula **anclas absolutas**: qué
    puntaje tendrían las imágenes más nítidas del propio conjunto si estuvieran desenfocadas
    un poco, bastante o mucho. Así el corte se puede pensar como "aceptar hasta un desenfoque
    leve" y no como "descartar el 10% peor", que descartaría lo mismo en un conjunto bueno o malo.
    """
    paths = [str(p) for p in paths]
    metrics = measure_many(paths, normalize_side, workers)
    ok = [(p, m.blur) for p, m in zip(paths, metrics, strict=True) if m is not None]
    if not ok:
        raise ValueError("No se pudo leer ninguna imagen")
    ok.sort(key=lambda item: item[1])
    blur = np.array([b for _, b in ok])

    reference = [p for p, _ in ok[-max(1, int(len(ok) * _REFERENCE_FRACTION)):]]
    grays = [
        cv2.cvtColor(np.asarray(load_image(p, max_side=normalize_side)), cv2.COLOR_RGB2GRAY)
        for p in reference
    ]
    anchors = {
        sigma: float(np.median([sharpness(cv2.GaussianBlur(g, (0, 0), sigma)) for g in grays]))
        for sigma in _ANCHOR_SIGMAS
    }
    return {
        "count": len(ok),
        "percentiles": {p: float(np.percentile(blur, p)) for p in _PERCENTILES},
        "anchors": anchors,
        "suggested": {"blur_min": anchors[2.0], "blur_ok": anchors[1.0]},
        "sorted_paths": [p for p, _ in ok],
        "sorted_blur": blur.tolist(),
    }
