"""Prueba de humo del servidor MCP del detector.

Lanza el servidor por stdio, lo llama como cliente MCP y compara el resultado con
una llamada directa al backend. Debe coincidir exactamente.

Uso:
    python -m eval.mcp_smoke --project projects/tomates-gold
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from autolabel.config import load_project
from autolabel.detectors import Query, create_detector
from autolabel.imaging import load_image


async def call_server(image_path: str, queries: list[dict], model_id: str, box_thr: float) -> dict:
    params = StdioServerParameters(command=sys.executable, args=["-m", "services.mcp_detector"])
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        result = await session.call_tool("detect", {
            "image_path": image_path, "queries": queries, "model_id": model_id,
            "box_threshold": box_thr, "text_threshold": box_thr,
        })
        if result.isError:
            raise RuntimeError(result.content)
        return {"tools": tools, "result": result.structuredContent or json.loads(result.content[0].text)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    args = ap.parse_args()

    cfg = load_project(args.project)
    gold = json.loads((cfg.root / "gold.json").read_text(encoding="utf-8"))
    image = cfg.images_dir / next(iter(gold))
    queries = [{"name": c.name, "prompt": c.prompt} for c in cfg.classes]
    thr = cfg.detector.box_threshold

    served = asyncio.run(call_server(str(image), queries, cfg.detector.model_id, thr))
    print("herramientas:", served["tools"])
    via_mcp = served["result"]["detections"]

    direct = create_detector(cfg.detector.backend, cfg.detector.model_id).detect(
        load_image(image), [Query(**q) for q in queries],
        box_threshold=thr, text_threshold=thr, nms_iou=cfg.detector.nms_iou,
    )
    elapsed = served["result"]["elapsed_ms"]
    print(f"detecciones por MCP: {len(via_mcp)}  directas: {len(direct)}  ({elapsed:.0f} ms)")
    same = len(via_mcp) == len(direct) and all(
        a["cls"] == b.cls and abs(a["score"] - b.score) < 1e-4
        for a, b in zip(via_mcp, direct, strict=True)
    )
    print("coinciden:", same)
    if not same:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
