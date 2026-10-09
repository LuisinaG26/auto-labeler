"""Grafo de LangGraph: filtro de calidad, curador, bucle etiquetado y verificación, salidas.

```
quality -> curate -> label <-> verify -> decide --(hay dudosas y quedan estrategias)--> label
                                          \\--(si no)--> finalize
```

Los pasos largos (etiquetar, verificar) procesan un lote por vez y vuelven a sí mismos hasta
terminar, así cada lote queda guardado en el checkpoint y una interrupción pierde poco. La
lógica de decisión vive en ``autolabel.loop`` y no sabe nada de LangGraph.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from autolabel.calibration import Calibration
from autolabel.config import ProjectConfig, StrategyConfig
from autolabel.curation import curate_images, save_curation
from autolabel.detectors import Query
from autolabel.export import write_yolo
from autolabel.imaging import list_images, load_image
from autolabel.labeling import AsyncDetector, drop_tiny_boxes
from autolabel.loop import (
    ImageState,
    accepted_labels,
    decide_image,
    merge_detections,
    verify_pending,
)
from autolabel.quality import assess_batch, save_assessments
from autolabel.reporting import detections_sheet
from autolabel.verification import Verifier
from autolabel.vlm import VisionLanguageModel

log = logging.getLogger("autolabel.graph")

DetectorFactory = Callable[[StrategyConfig, list[Query]], AbstractAsyncContextManager[AsyncDetector]]


class RunState(TypedDict, total=False):
    images: dict[str, dict]       # nombre -> ImageState como diccionario
    quality: dict                 # resumen del filtro
    todo_label: list[str]         # imágenes que faltan etiquetar en esta ronda
    todo_verify: list[str]        # imágenes etiquetadas que faltan verificar
    active: list[str]             # imágenes que participan en la ronda actual
    round: int                    # índice de la estrategia de la ronda actual
    elapsed: float                # segundos de trabajo acumulados, sobrevive a las reanudaciones
    history: list[dict]           # resumen por ronda
    done: bool
    summary: dict


@dataclass
class Deps:
    """Todo lo que el grafo necesita del exterior, para poder sustituirlo en las pruebas."""

    cfg: ProjectConfig
    open_detector: DetectorFactory
    verifier: Verifier
    calibration: Calibration | None
    queries: list[Query]
    names: Sequence[str] | None = None            # restringe a estas imágenes
    vlm: VisionLanguageModel | None = None        # para el curador
    curate: bool = True
    chunk_label: int = 40
    chunk_verify: int = 10


REVIEW_PREVIEW_IMAGES = 24
_CONFIRMED_COLOR, _DOUBTFUL_COLOR = (0, 200, 0), (255, 215, 0)   # confirmada y dudosa


def _review_preview(
    cfg: ProjectConfig, images: dict[str, ImageState], names: list[str], out: Path
) -> None:
    """Hoja con las imágenes a revisar, con distinto color para lo confirmado y para lo dudoso."""
    items = []
    for name in names:
        s = images[name]
        boxes = [
            (b.box, f"{(b.cls or '-')[:4]} {b.confidence:.2f}",
             _CONFIRMED_COLOR if b.status == "accepted" else _DOUBTFUL_COLOR)
            for b in s.boxes if b.status in ("accepted", "uncertain")
        ]
        items.append((cfg.images_dir / name, s.width, s.height, boxes))
    detections_sheet(items, out, columns=6, thumb=300)


def build_graph(deps: Deps) -> StateGraph:
    cfg = deps.cfg
    strategies = cfg.strategies()
    run_dir = cfg.out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ nodos

    def quality(_: RunState) -> dict:
        wanted = set(deps.names) if deps.names is not None else None
        paths = [p for p in list_images(cfg.images_dir) if wanted is None or p.name in wanted]
        results = assess_batch(paths, cfg.quality)
        save_assessments(run_dir / "quality.json", cfg.quality.model_dump(), results)
        by_verdict: dict[str, list[str]] = {"keep": [], "unsure": [], "drop": []}
        for a in results:
            by_verdict[a.verdict].append(Path(a.image).name)
        log.info("Calidad: %s", {k: len(v) for k, v in by_verdict.items()})
        return {"quality": {k: sorted(v) for k, v in by_verdict.items()}}

    def curate(state: RunState) -> dict:
        q = state["quality"]
        selected = list(q["keep"])
        if q["unsure"] and deps.curate and deps.vlm is not None:
            results = curate_images(cfg, deps.vlm, q["unsure"])
            save_curation(run_dir / "curation.json", results)
            kept = [r.image for r in results if r.verdict == "keep"]
            log.info("Curador: %d de %d dudosas pasan", len(kept), len(results))
            selected += kept
        selected = sorted(set(selected))
        return {
            "images": {n: ImageState(n).to_dict() for n in selected},
            "active": selected, "todo_label": selected, "todo_verify": [],
            "round": 0, "elapsed": 0.0, "history": [],
        }

    def label(state: RunState) -> dict:
        start = time.perf_counter()
        r = state["round"]
        strategy = strategies[r]
        chunk, rest = state["todo_label"][: deps.chunk_label], state["todo_label"][deps.chunk_label:]
        images = dict(state["images"])

        async def run() -> None:
            async with deps.open_detector(strategy, deps.queries) as detector:
                for name in chunk:
                    out = await detector.detect(cfg.images_dir / name)
                    st = ImageState.from_dict(images[name])
                    st.width, st.height = out.width, out.height
                    merge_detections(st, drop_tiny_boxes(cfg, out.detections), r, strategy.model_id)
                    images[name] = st.to_dict()

        asyncio.run(run())
        log.info("Ronda %d, etiquetadas %d imágenes (quedan %d)", r + 1, len(chunk), len(rest))
        return {"images": images, "todo_label": rest, "todo_verify": state["todo_verify"] + chunk,
                "elapsed": state["elapsed"] + time.perf_counter() - start}

    def verify(state: RunState) -> dict:
        start = time.perf_counter()
        chunk, rest = state["todo_verify"][: deps.chunk_verify], state["todo_verify"][deps.chunk_verify:]
        images = dict(state["images"])
        for name in chunk:
            st = ImageState.from_dict(images[name])
            if st.boxes:
                verify_pending(st, load_image(cfg.images_dir / name), deps.verifier, deps.calibration)
            images[name] = st.to_dict()
        log.info("Verificadas %d imágenes (quedan %d)", len(chunk), len(rest))
        elapsed = state["elapsed"] + time.perf_counter() - start
        return {"images": images, "todo_verify": rest, "elapsed": elapsed}

    def decide(state: RunState) -> dict:
        r = state["round"]
        budget = cfg.labeling.time_budget_min
        out_of_time = budget is not None and state["elapsed"] > budget * 60
        can_retry = r < len(strategies) - 1 and not out_of_time

        images = dict(state["images"])
        next_active: list[str] = []
        for name in state["active"]:
            st = ImageState.from_dict(images[name])
            if decide_image(st, cfg, can_retry=can_retry, min_gain=cfg.labeling.min_gain) == "active":
                next_active.append(name)
            images[name] = st.to_dict()

        counts = Counter(img["status"] for img in images.values())
        entry = {"round": r, "model": strategies[r].model_id, "tiles": strategies[r].tiles,
                 "processed": len(state["active"]), "next_active": len(next_active),
                 "accepted": counts["accepted"], "review": counts["review"],
                 "elapsed_s": round(state["elapsed"], 1)}
        (run_dir / f"round_{r}.json").write_text(json.dumps(images), encoding="utf-8")
        log.info("Ronda %d lista: %s", r + 1, {k: entry[k] for k in ("accepted", "review", "next_active")})
        return {"images": images, "active": next_active, "todo_label": next_active, "round": r + 1,
                "history": [*state["history"], entry]}

    def finalize(state: RunState) -> dict:
        images = {n: ImageState.from_dict(d) for n, d in state["images"].items()}
        accepted = [accepted_labels(s) for s in images.values() if s.status == "accepted"]
        write_yolo(accepted, cfg.class_names, cfg.images_dir, run_dir / "yolo")
        to_review = [s for s in images.values() if s.status == "review"]
        # Las imágenes con más cajas dudosas primero: ahí está el trabajo
        review = sorted(
            ({"image": s.name, "reason": s.reason, **{k: sum(b.status == k for b in s.boxes)
                                                       for k in ("accepted", "uncertain", "rejected")}}
             for s in to_review),
            key=lambda r: (-r["uncertain"], r["image"]),
        )
        (run_dir / "review.json").write_text(
            json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        if review:
            _review_preview(cfg, images, [r["image"] for r in review[:REVIEW_PREVIEW_IMAGES]],
                            run_dir / "review_preview.jpg")

        boxes = Counter(b.status for s in images.values() for b in s.boxes)
        summary = {
            "images": dict(Counter(s.status for s in images.values())),
            "boxes": dict(boxes),
            "dropped_by_quality": len(state["quality"]["drop"]),
            "rounds": len(state["history"]),
            "elapsed_s": round(state["elapsed"], 1),
        }
        (run_dir / "final.json").write_text(json.dumps({
            "summary": summary, "history": state["history"], "images": [s.to_dict() for s in images.values()],
        }), encoding="utf-8")
        log.info("Resultado: %s", summary)
        return {"done": True, "summary": summary}

    # ------------------------------------------------------------------ rutas

    def after_label(state: RunState) -> str:
        return "label" if state["todo_label"] else "verify"

    def after_verify(state: RunState) -> str:
        return "verify" if state["todo_verify"] else "decide"

    def after_decide(state: RunState) -> str:
        return "label" if state["active"] else "finalize"

    graph = StateGraph(RunState)
    for name, fn in [("quality", quality), ("curate", curate), ("label", label), ("verify", verify),
                     ("decide", decide), ("finalize", finalize)]:
        graph.add_node(name, fn)
    graph.add_edge(START, "quality")
    graph.add_edge("quality", "curate")
    graph.add_edge("curate", "label")
    graph.add_conditional_edges("label", after_label, {"label": "label", "verify": "verify"})
    graph.add_conditional_edges("verify", after_verify, {"verify": "verify", "decide": "decide"})
    graph.add_conditional_edges("decide", after_decide, {"label": "label", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph


def run_graph(deps: Deps, *, restart: bool = False, thread_id: str = "run") -> dict[str, Any]:
    """Corre el grafo o retoma la última corrida interrumpida. Devuelve el estado final."""
    run_dir = deps.cfg.out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    database = run_dir / "checkpoints.sqlite"
    if restart and database.exists():
        database.unlink()

    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_graph(deps).compile(checkpointer=saver)
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100_000}
        snapshot = graph.get_state(config)
        if snapshot.next:
            log.info("Retomando una corrida interrumpida")
            return graph.invoke(None, config)
        if snapshot.values.get("done"):
            log.info("La corrida ya estaba terminada: usa --restart para repetirla")
            return dict(snapshot.values)
        for stale in run_dir.glob("round_*.json"):      # fotos de una corrida anterior: confunden
            stale.unlink()
        return graph.invoke({}, config)
