"""Acceso al detector: por MCP (arquitectura normal) o en el mismo proceso (sin servidor).

Ambos adaptadores exponen lo mismo: ``await detector.detect(path)`` devuelve el tamaño de la
imagen y las detecciones, con coordenadas sobre la imagen ya girada según EXIF.
"""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from autolabel.config import StrategyConfig
from autolabel.detectors import Detection, Query


@dataclass(frozen=True, slots=True)
class DetectOutput:
    width: int
    height: int
    detections: tuple[Detection, ...]


class McpDetector:
    """Cliente del servidor MCP del detector. Mantiene una sola sesión para todo el lote."""

    def __init__(self, url: str, queries: list[Query], cfg: StrategyConfig) -> None:
        self._url = url
        self._queries = [{"name": q.name, "prompt": q.prompt} for q in queries]
        self._cfg = cfg
        self._stack = AsyncExitStack()
        self._session = None

    async def __aenter__(self) -> Self:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        read, write, _ = await self._stack.enter_async_context(streamablehttp_client(self._url))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def detect(self, path: Path) -> DetectOutput:
        assert self._session is not None, "usar dentro de 'async with'"
        result = await self._session.call_tool(
            "detect",
            {
                "image_path": str(path),
                "queries": self._queries,
                "backend": self._cfg.backend,
                "model_id": self._cfg.model_id,
                "box_threshold": self._cfg.box_threshold,
                "text_threshold": self._cfg.text_threshold,
                "nms_iou": self._cfg.nms_iou,
                "tiles": self._cfg.tiles,
            },
        )
        if result.isError:
            raise RuntimeError(f"El detector falló en {path.name}: {result.content}")
        data = result.structuredContent or json.loads(result.content[0].text)
        width, height = data["image_size"]
        dets = tuple(Detection(d["cls"], tuple(d["box"]), d["score"]) for d in data["detections"])
        return DetectOutput(width, height, dets)


class LocalDetector:
    """Detector en el mismo proceso. Requiere la imagen con GPU (torch y transformers)."""

    def __init__(self, queries: list[Query], cfg: StrategyConfig) -> None:
        from autolabel.detectors import create_detector

        self._queries = queries
        self._cfg = cfg
        self._detector = create_detector(cfg.backend, cfg.model_id)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def detect(self, path: Path) -> DetectOutput:
        from autolabel.imaging import load_image

        img = load_image(path)
        dets = self._detector.detect(
            img, self._queries,
            box_threshold=self._cfg.box_threshold,
            text_threshold=self._cfg.text_threshold,
            nms_iou=self._cfg.nms_iou,
            tiles=self._cfg.tiles,
        )
        return DetectOutput(img.width, img.height, tuple(dets))
