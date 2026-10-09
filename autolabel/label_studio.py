"""Integración con Label Studio: tareas con preanotaciones y cliente de su API.

Las cajas del detector se cargan como predicciones, así el humano corrige en vez de dibujar.
Label Studio usa porcentajes (0 a 100) relativos a la imagen ya girada según EXIF.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import PurePosixPath
from xml.sax.saxutils import quoteattr

import httpx

from autolabel.geometry import clip_boxes
from autolabel.labeling import ImageLabels

LABEL_FROM, IMAGE_TO = "label", "image"


def labeling_config(class_names: Sequence[str]) -> str:
    """Plantilla de etiquetado: una imagen y una etiqueta rectangular por clase."""
    labels = "\n".join(f"    <Label value={quoteattr(n)}/>" for n in class_names)
    return (
        f'<View>\n  <Image name="{IMAGE_TO}" value="$image"/>\n'
        f'  <RectangleLabels name="{LABEL_FROM}" toName="{IMAGE_TO}">\n{labels}\n'
        "  </RectangleLabels>\n</View>"
    )


def _result(item: ImageLabels, box: Sequence[float], cls: str, score: float) -> dict | None:
    x1, y1, x2, y2 = clip_boxes([box], item.width, item.height)[0]
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "from_name": LABEL_FROM,
        "to_name": IMAGE_TO,
        "type": "rectanglelabels",
        "original_width": item.width,
        "original_height": item.height,
        "image_rotation": 0,
        "score": round(score, 4),
        "value": {
            "x": x1 / item.width * 100,
            "y": y1 / item.height * 100,
            "width": (x2 - x1) / item.width * 100,
            "height": (y2 - y1) / item.height * 100,
            "rotation": 0,
            "rectanglelabels": [cls],
        },
    }


def build_tasks(
    results: Sequence[ImageLabels],
    image_url: Callable[[str], str],
    model_version: str,
    extra: Callable[[str], dict] | None = None,
) -> list[dict]:
    """Una tarea por imagen con sus cajas como predicción. Las sin detecciones van sin predicción.

    ``extra`` agrega campos de datos por imagen, que Label Studio muestra como columnas y
    permite ordenar y filtrar, por ejemplo cuántas cajas dudosas tiene cada una. Puede devolver
    ``None`` para una imagen sin datos extra.
    """
    tasks = []
    for item in results:
        boxes = [r for d in item.detections if (r := _result(item, d.box, d.cls, d.score))]
        data = {"image": image_url(item.image), **((extra(item.image) if extra else None) or {})}
        task: dict = {"data": data, "meta": {"file": item.image}}
        if boxes:
            task["predictions"] = [{
                "model_version": model_version,
                "score": round(sum(b["score"] for b in boxes) / len(boxes), 4),
                "result": boxes,
            }]
        tasks.append(task)
    return tasks


def local_file_url(path: str, document_root: str) -> str:
    """URL con la que Label Studio sirve un archivo local (ver LOCAL_FILES_DOCUMENT_ROOT)."""
    rel = PurePosixPath(path).relative_to(document_root)
    return f"/data/local-files/?d={rel}"


class LabelStudio:
    """Cliente mínimo de la API REST."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """``token`` es un token heredado de la API. Las versiones recientes lo deshabilitan por
        defecto: en ese caso usar ``login`` con usuario y contraseña."""
        headers = {"Authorization": f"Token {token}"} if token else {}
        self._http = httpx.Client(
            base_url=url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )

    def login(self, username: str, password: str) -> None:
        """Inicia sesión como lo haría el navegador y deja la cookie y el token CSRF listos."""
        self._http.get("/user/login/").raise_for_status()
        response = self._http.post(
            "/user/login/",
            data={
                "email": username,
                "password": password,
                "csrfmiddlewaretoken": self._http.cookies.get("csrftoken", ""),
            },
        )
        # Un inicio de sesión correcto redirige. Si devuelve el formulario de nuevo, falló.
        if response.status_code not in (301, 302, 303) and "sessionid" not in self._http.cookies:
            raise PermissionError("Label Studio rechazó el usuario o la contraseña")
        self._http.headers["X-CSRFToken"] = self._http.cookies.get("csrftoken", "")

    def wait_ready(self, seconds: float = 120.0) -> None:
        deadline = time.monotonic() + seconds
        while True:
            try:
                if self._http.get("/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Label Studio no respondió a tiempo")
            time.sleep(2)

    def ensure_project(self, title: str, config: str) -> int:
        """Devuelve el id del proyecto con ese título. Lo crea si no existe y actualiza su plantilla."""
        r = self._http.get("/api/projects", params={"title": title, "page_size": 100})
        r.raise_for_status()
        for p in r.json().get("results", []):
            if p["title"] == title:
                self._http.patch(f"/api/projects/{p['id']}", json={"label_config": config}).raise_for_status()
                return p["id"]
        r = self._http.post("/api/projects", json={"title": title, "label_config": config})
        r.raise_for_status()
        return r.json()["id"]

    def ensure_local_storage(self, project_id: int, path: str) -> None:
        """Registra la carpeta de imágenes: sin esto Label Studio no sirve archivos locales."""
        r = self._http.get("/api/storages/localfiles", params={"project": project_id})
        r.raise_for_status()
        if any(s["path"] == path for s in r.json()):
            return
        self._http.post(
            "/api/storages/localfiles",
            json={"project": project_id, "title": PurePosixPath(path).name, "path": path,
                  "use_blob_urls": True},
        ).raise_for_status()

    def task_count(self, project_id: int) -> int:
        r = self._http.get(f"/api/projects/{project_id}")
        r.raise_for_status()
        return r.json().get("task_number", 0)

    def delete_tasks(self, project_id: int) -> None:
        self._http.delete(f"/api/projects/{project_id}/tasks").raise_for_status()

    def import_tasks(self, project_id: int, tasks: list[dict]) -> int:
        r = self._http.post(
            f"/api/projects/{project_id}/import", params={"return_task_ids": "true"}, json=tasks
        )
        r.raise_for_status()
        return r.json().get("task_count", len(tasks))
