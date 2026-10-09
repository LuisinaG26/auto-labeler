"""Backend Grounding DINO (Hugging Face transformers)."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from autolabel.detectors.base import Detection, Query
from autolabel.tiling import is_cut_by_tile, tile_grid


class GroundingDinoDetector:
    """Detector guiado por texto.

    Cada consulta es un elemento del lote: la misma imagen repetida con un prompt
    distinto. Así la clase de cada caja es inequívoca, sin reconstruir qué frase
    de un prompt combinado coincidió, y todo se resuelve en una sola pasada.
    """

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            .to(self.device)
            .eval()
        )

    def detect(
        self,
        image: Image.Image,
        queries: Sequence[Query],
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        nms_iou: float = 0.5,
        max_dets: int = 300,
        tiles: int = 1,
    ) -> list[Detection]:
        """Con ``tiles > 1`` se detecta además en cada tile y se funden los resultados.

        Se conserva la pasada sobre la imagen completa, que ve bien los objetos grandes. Las
        cajas que un tile parte por su borde interior se descartan: el tile vecino las ve enteras.
        """
        if not queries:
            return []
        image = image.convert("RGB")
        found = self._detect_region(image, queries, box_threshold, text_threshold)

        if tiles > 1:
            for tile in tile_grid(image.width, image.height, tiles):
                ox, oy = tile[0], tile[1]
                for d in self._detect_region(image.crop(tile), queries, box_threshold, text_threshold):
                    box = (d.box[0] + ox, d.box[1] + oy, d.box[2] + ox, d.box[3] + oy)
                    if not is_cut_by_tile(box, tile, image.size):
                        found.append(Detection(d.cls, box, d.score))

        return self._merge(found, nms_iou, max_dets)

    def _detect_region(
        self, region: Image.Image, queries: Sequence[Query], box_threshold: float, text_threshold: float
    ) -> list[Detection]:
        """Una pasada del modelo sobre una región, con todas las consultas en un lote."""
        n = len(queries)
        with torch.inference_mode():
            inputs = self.processor(
                images=[region] * n,
                text=[[q.prompt.lower()] for q in queries],
                padding="longest",
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[(region.height, region.width)] * n,
            )

        detections: list[Detection] = []
        for query, result in zip(queries, results, strict=True):
            detections.extend(
                Detection(query.name, tuple(box), score)
                for box, score in zip(result["boxes"].tolist(), result["scores"].tolist(), strict=True)
            )
        return detections

    @staticmethod
    def _merge(detections: list[Detection], nms_iou: float, max_dets: int) -> list[Detection]:
        """Supresión de no máximos por clase, en una sola llamada."""
        if not detections:
            return []
        names = sorted({d.cls for d in detections})
        index = {n: i for i, n in enumerate(names)}
        boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
        scores = torch.tensor([d.score for d in detections], dtype=torch.float32)
        classes = torch.tensor([index[d.cls] for d in detections])
        keep = batched_nms(boxes, scores, classes, nms_iou)[:max_dets]
        return [detections[i] for i in keep.tolist()]
