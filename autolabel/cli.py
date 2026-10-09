"""Línea de comandos de autolabel. Los subcomandos se suman con cada fase.

    python -m autolabel.cli quality     --project projects/mi-proyecto
    python -m autolabel.cli label       --project projects/mi-proyecto
    python -m autolabel.cli labelstudio --project projects/mi-proyecto
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from autolabel.config import ProjectConfig, load_project
from autolabel.imaging import list_images
from autolabel.quality import Assessment, assess_batch, save_assessments
from autolabel.quality import calibration_report as quality_calibration_report
from autolabel.reporting import GREEN, contact_sheet, detections_sheet

log = logging.getLogger("autolabel")

DEFAULT_DETECTOR_URL = "http://mcp-detector:8001/mcp"
DEFAULT_LABEL_STUDIO_URL = "http://label-studio:8080"
DEFAULT_LS_DOCUMENT_ROOT = "/data"
# Rutas de este contenedor -> rutas con las que Label Studio ve los mismos archivos
DEFAULT_PATH_MAP = ("/app/projects=/data/projects",)


# ----------------------------------------------------------------------- init


def run_init(args: argparse.Namespace) -> None:
    from autolabel.scaffold import init_project

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    try:
        target = init_project(args.folder, classes, images=args.images, scene=args.scene, force=args.force)
    except (FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from e
    log.info("Proyecto creado: %s", target)
    log.info("Siguientes pasos: poné las imágenes en %s", args.images)
    log.info("y completá el prompt y la definición de cada clase.")
    log.info("Después: python -m autolabel.cli quality --project %s --calibrate", Path(args.folder))


# ----------------------------------------------------------------------- quality


def run_quality_calibration(cfg: ProjectConfig, sheet_size: int = 30) -> None:
    """Sugiere cortes de nitidez y deja una hoja de muestras a lo largo de la distribución."""
    images = list_images(cfg.images_dir)
    if not images:
        raise SystemExit(f"No hay imágenes en {cfg.images_dir}")
    report = quality_calibration_report(images, cfg.quality.normalize_side)

    log.info("Imágenes medidas: %d", report["count"])
    log.info("Nitidez, percentiles: %s", {p: round(v) for p, v in report["percentiles"].items()})
    log.info("Anclas absolutas, o sea el puntaje de las imágenes más nítidas del conjunto con desenfoque")
    for sigma, value in report["anchors"].items():
        log.info("  gaussiano de %.0f px: %.0f", sigma, value)
    suggested = report["suggested"]
    log.info("Cortes sugeridos: blur_min=%.0f (equivale a desenfoque 2) y blur_ok=%.0f (desenfoque 1)",
             suggested["blur_min"], suggested["blur_ok"])

    n = len(report["sorted_paths"])
    picks = sorted({round(i * (n - 1) / (sheet_size - 1)) for i in range(sheet_size)})
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    sheet = cfg.out_dir / "quality_quantiles.jpg"
    contact_sheet(
        [report["sorted_paths"][i] for i in picks],
        [f"{report['sorted_blur'][i]:.0f} p{100 * i // n}" for i in picks],
        sheet, columns=6, thumb=260,
    )
    log.info("Hoja de muestras, de la menos a la más nítida: %s", sheet)
    log.info("Confirmá a ojo desde qué puntaje las imágenes dejan de servir y copiá los cortes a "
             "quality.blur_min y quality.blur_ok de project.yaml")


def run_quality(cfg: ProjectConfig, review: int) -> list[Assessment]:
    images = list_images(cfg.images_dir)
    if not images:
        raise SystemExit(f"No hay imágenes en {cfg.images_dir}")
    if cfg.quality.blur_min is None:
        log.warning("quality.blur_min y blur_ok no están definidos: no se filtra por desenfoque. "
                    "Para fijarlos: python -m autolabel.cli quality --project %s --calibrate", cfg.root)
    log.info("Evaluando %d imágenes de %s", len(images), cfg.images_dir)
    results = assess_batch(images, cfg.quality)

    out = cfg.out_dir / "quality.json"
    save_assessments(out, cfg.quality.model_dump(), results)

    counts = Counter(a.verdict for a in results)
    reasons = Counter(r.split(":")[0] for a in results for r in a.reasons)
    log.info("Resultado: %s", dict(counts))
    log.info("Motivos: %s", dict(reasons))
    log.info("Guardado en %s", out)

    if review:
        scored = sorted((a for a in results if a.metrics), key=lambda a: a.metrics.blur)[:review]
        sheet = cfg.out_dir / "quality_lowest_blur.jpg"
        contact_sheet(
            [a.image for a in scored],
            [f"{a.metrics.blur:.0f} {a.verdict}" for a in scored],
            sheet,
        )
        log.info("Hoja con las %d imágenes menos nítidas: %s", len(scored), sheet)
    return results


# ----------------------------------------------------------------------- label


def run_label(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    from autolabel.detector_client import LocalDetector, McpDetector
    from autolabel.export import write_yolo
    from autolabel.labeling import build_queries, label_images, select_images

    paths = select_images(cfg, include_unsure=args.include_unsure, limit=args.limit, seed=args.seed)
    if not paths:
        raise SystemExit("Ninguna imagen pasó el filtro de calidad")
    queries = build_queries(cfg, args.mode)

    async def run():
        if args.local:
            detector_cm = LocalDetector(queries, cfg.detector)
        else:
            detector_cm = McpDetector(args.server, queries, cfg.detector)
        async with detector_cm as detector:
            return await label_images(cfg, detector, paths, queries, args.mode)

    results = asyncio.run(run())

    class_names = [q.name for q in queries]
    yolo_dir = write_yolo(results, class_names, cfg.images_dir, cfg.out_dir / "yolo")

    counts = np.array([len(r.detections) for r in results])
    scores = np.array([d.score for r in results for d in r.detections])
    log.info("Imágenes etiquetadas: %d", len(results))
    log.info("Cajas: %d en total, %.1f por imagen, %d imágenes sin ninguna", counts.sum(), counts.mean(),
             int((counts == 0).sum()))
    if len(scores):
        log.info("Score de las cajas, percentiles 10/50/90: %s", np.percentile(scores, [10, 50, 90]).round(2))
    log.info("YOLO: %s", yolo_dir)

    if args.preview:
        sample = random.Random(args.seed).sample(results, min(args.preview, len(results)))
        sheet = cfg.out_dir / "labels_preview.jpg"
        detections_sheet(
            [(cfg.images_dir / r.image, r.width, r.height,
              [(d.box, f"{d.score:.2f}", GREEN) for d in r.detections]) for r in sample],
            sheet,
        )
        log.info("Vista previa con cajas: %s", sheet)


# ----------------------------------------------------------------------- curate


def run_curate(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    from autolabel.curation import curate_images, save_curation
    from autolabel.vlm import create_vlm

    quality_file = cfg.out_dir / "quality.json"
    if not quality_file.is_file():
        raise SystemExit(
            f"Falta {quality_file}. Corre antes: python -m autolabel.cli quality --project {cfg.root}"
        )
    records = json.loads(quality_file.read_text(encoding="utf-8"))["images"]
    names = sorted(r["image"] for r in records if r["verdict"] == "unsure")
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit("No hay imágenes dudosas que curar")
    log.info("Curando %d imágenes dudosas con %s", len(names), cfg.vlm.model)
    results = curate_images(cfg, create_vlm(cfg.vlm), names)
    save_curation(cfg.out_dir / "curation.json", results)

    counts = Counter(r.verdict for r in results)
    log.info("Resultado: %s", dict(counts))
    for r in [r for r in results if r.verdict != "keep"][:5]:
        log.info("  %s -> %s: %s", r.image, r.verdict, r.reason)


# ----------------------------------------------------------------------- verify

_CLASS_COLORS = [(0, 200, 0), (255, 215, 0), (0, 150, 255), (200, 0, 200), (0, 220, 220), (255, 120, 180)]
_NOT_AN_OBJECT, _GROUP = (255, 0, 0), (255, 128, 0)


def _load_calibration(cfg: ProjectConfig, path: str | None):
    """Calibración del proyecto, o ``None`` si no existe: entonces la confianza es el score del detector."""
    from autolabel.calibration import Calibration

    calibration_path = Path(path) if path else cfg.out_dir / "calibration.json"
    if not calibration_path.is_file():
        log.warning("Sin calibración (%s): la confianza será el score del detector", calibration_path)
        return None
    return Calibration.load(calibration_path)


def _reference_sheet(cfg: ProjectConfig, disabled: bool):
    """Hoja con las referencias de cada clase, o ``None`` si no hay o se pidió no usarlas."""
    from autolabel.references import load_references, reference_sheet

    if disabled:
        return None
    refs = load_references(cfg)
    return reference_sheet(refs, list(cfg.class_names)) if refs else None


def run_verify(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    from autolabel.labeling import load_labels
    from autolabel.verifying import verify_labels
    from autolabel.vlm import create_vlm

    labels_file = cfg.out_dir / "labels.json"
    if not labels_file.is_file():
        raise SystemExit(
            f"Falta {labels_file}. Corre antes: python -m autolabel.cli label --project {cfg.root}"
        )
    _, labels = load_labels(labels_file)
    if args.limit:
        labels = labels[: args.limit]

    calibration = _load_calibration(cfg, args.calibration)
    sheet = _reference_sheet(cfg, args.no_refs)
    log.info("Verificando %d imágenes con %s%s", len(labels), cfg.vlm.model,
             " y referencias" if sheet is not None else "")
    results = verify_labels(cfg, create_vlm(cfg.vlm), labels, calibration=calibration, reference_sheet=sheet)

    boxes = [b for r in results for b in r.boxes]
    if not boxes:
        log.info("No hay cajas que verificar")
        return
    conf = np.array([b.confidence for b in boxes])
    log.info("Cajas verificadas: %d", len(boxes))
    log.info("No parecen un objeto (p_none>=0.5): %d", sum(b.verdict.p_none >= 0.5 for b in boxes))
    log.info("Parecen un grupo (p_several>=0.5): %d", sum(b.verdict.p_several >= 0.5 for b in boxes))
    log.info("Clase según el VLM: %s", dict(Counter(b.cls or "ninguna" for b in boxes)))
    log.info("Confianza, percentiles 10/50/90: %s", np.percentile(conf, [10, 50, 90]).round(2))

    if args.preview:
        index = {name: i for i, name in enumerate(cfg.class_names)}
        sample = random.Random(args.seed).sample(results, min(args.preview, len(results)))

        def style(b):
            v = b.verdict
            if v.p_none >= 0.5:
                return _NOT_AN_OBJECT
            if v.p_several >= 0.5:
                return _GROUP
            return _CLASS_COLORS[index.get(b.cls, 0) % len(_CLASS_COLORS)] if b.cls else _NOT_AN_OBJECT

        sheet_path = cfg.out_dir / "verification_preview.jpg"
        detections_sheet(
            [(cfg.images_dir / r.image, r.width, r.height,
              [(b.verdict.box, f"{(b.cls or '-')[:3]} {b.confidence:.2f}", style(b)) for b in r.boxes])
             for r in sample],
            sheet_path,
        )
        log.info("Vista previa (rojo: no es objeto, naranja: grupo, otros colores: clase): %s", sheet_path)


# ----------------------------------------------------------------------- run


def run_pipeline(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    from autolabel.detector_client import LocalDetector, McpDetector
    from autolabel.graph import Deps, run_graph
    from autolabel.labeling import build_queries
    from autolabel.verification import Verifier
    from autolabel.vlm import create_vlm

    if args.rounds:                                  # menos rondas que las del YAML, para comparar
        limited = cfg.labeling.model_copy(update={"max_iterations": args.rounds})
        cfg = cfg.model_copy(update={"labeling": limited})
    names = None
    if args.names_file:
        lines = Path(args.names_file).read_text(encoding="utf-8").splitlines()
        names = sorted(n.strip() for n in lines if n.strip())
    elif args.limit:
        all_names = [p.name for p in list_images(cfg.images_dir)]
        names = sorted(random.Random(args.seed).sample(all_names, min(args.limit, len(all_names))))

    queries = build_queries(cfg, cfg.labeling.mode)
    vlm = create_vlm(cfg.vlm)
    sheet = _reference_sheet(cfg, args.no_refs)
    calibration = _load_calibration(cfg, args.calibration)

    def open_detector(strategy, strategy_queries):
        if args.local:
            return LocalDetector(strategy_queries, strategy)
        return McpDetector(args.server, strategy_queries, strategy)

    strategies = cfg.strategies()
    log.info("Estrategias: %s", [f"{s.model_id.rsplit('/', 1)[-1]} umbral {s.box_threshold} tiles {s.tiles}"
                                 for s in strategies])
    deps = Deps(cfg, open_detector, Verifier(cfg, vlm, sheet), calibration, queries, names=names,
                vlm=vlm, curate=not args.no_curate)
    state = run_graph(deps, restart=args.restart)

    summary = state.get("summary", {})
    log.info("Imágenes: %s", summary.get("images"))
    log.info("Cajas: %s", summary.get("boxes"))
    log.info("Apartadas por calidad: %s | rondas: %s | %.0f s de trabajo", summary.get("dropped_by_quality"),
             summary.get("rounds"), summary.get("elapsed_s", 0))
    log.info("Resultados en %s", cfg.out_dir / "run")


# ----------------------------------------------------------------------- labelstudio


def _map_path(path: str, path_map: list[str]) -> str:
    for rule in path_map:
        src, dst = rule.split("=", 1)
        if path.startswith(src):
            return dst + path[len(src):]
    return path


def run_labelstudio(cfg: ProjectConfig, args: argparse.Namespace) -> None:
    from autolabel.label_studio import LabelStudio, build_tasks, labeling_config, local_file_url
    from autolabel.labeling import load_labels
    from autolabel.loop import ImageState, review_labels

    if args.from_run:
        final_file = cfg.out_dir / "run" / "final.json"
        if not final_file.is_file():
            raise SystemExit(
                f"Falta {final_file}. Corre antes: python -m autolabel.cli run --project {cfg.root}"
            )
        data = json.loads(final_file.read_text(encoding="utf-8"))
        states = [ImageState.from_dict(d) for d in data["images"]]
        in_review = [s for s in states if s.status == "review"]
        results = [review_labels(s) for s in in_review]
        meta = {"classes": list(cfg.class_names), "model_id": cfg.vlm.model, "box_threshold": "verificado"}
        # Columnas de cada tarea en Label Studio, para ordenar y filtrar por lo que hay que mirar
        counts = {
            s.name: {
                "uncertain_boxes": sum(b.status == "uncertain" for b in s.boxes),
                "confirmed_boxes": sum(b.status == "accepted" for b in s.boxes),
                "reason": s.reason,
            }
            for s in in_review
        }
        extra = counts.get
        # Un proyecto aparte del de las preanotaciones sueltas, para no mezclar tareas de orígenes distintos
        title = args.title or f"{cfg.name}-revision"
        log.info("%d imágenes en revisión de %d procesadas", len(results), len(states))
    else:
        extra, title = None, args.title or cfg.name
        labels_file = cfg.out_dir / "labels.json"
        if not labels_file.is_file():
            raise SystemExit(
                f"Falta {labels_file}. Corre antes: python -m autolabel.cli label --project {cfg.root}"
            )
        meta, results = load_labels(labels_file)

    images_dir = _map_path(str(cfg.images_dir), args.path_map)
    client = LabelStudio(args.url, args.token)
    client.wait_ready()
    if not args.token:
        client.login(args.username, args.password)
    project_id = client.ensure_project(title, labeling_config(meta["classes"]))
    client.ensure_local_storage(project_id, images_dir)

    existing = client.task_count(project_id)
    if existing and not args.replace:
        raise SystemExit(f"El proyecto ya tiene {existing} tareas. Usa --replace para reemplazarlas.")
    if existing:
        client.delete_tasks(project_id)

    version = f"{Path(meta['model_id']).name}@{meta['box_threshold']}"
    tasks = build_tasks(
        results, lambda name: local_file_url(f"{images_dir}/{name}", args.document_root), version, extra
    )
    n = client.import_tasks(project_id, tasks)
    log.info("Importadas %d tareas en el proyecto %d: %s/projects/%d", n, project_id, args.url, project_id)


# ----------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(prog="autolabel")
    sub = parser.add_subparsers(dest="command", required=True)

    ini = sub.add_parser("init", help="Crea la carpeta de un proyecto nuevo con su project.yaml")
    ini.add_argument("folder", help="Carpeta del proyecto, por ejemplo projects/mi-proyecto")
    ini.add_argument("--classes", required=True, help="Nombres de las clases separados por coma")
    ini.add_argument("--images", default="./images",
                     help="Carpeta de imágenes, relativa al proyecto o absoluta")
    ini.add_argument("--scene", default="", help="Frase que describe qué muestran las fotos")
    ini.add_argument("--force", action="store_true", help="Reemplaza un project.yaml existente")

    q = sub.add_parser("quality", help="Filtro de calidad de imágenes")
    q.add_argument("--project", required=True)
    q.add_argument("--review", type=int, default=0, metavar="N",
                   help="Genera una hoja con las N imágenes menos nítidas para confirmar el corte")
    q.add_argument("--calibrate", action="store_true",
                   help="Sugiere los cortes de nitidez y genera una hoja de muestras para confirmarlos")

    lb = sub.add_parser("label", help="Etiquetado automático por lotes y exportación a YOLO")
    lb.add_argument("--project", required=True)
    lb.add_argument("--mode", choices=["agnostic", "direct"], default="agnostic",
                    help="agnostic: un prompt para todo. direct: un prompt por clase")
    lb.add_argument("--server", default=os.environ.get("AUTOLABEL_DETECTOR_URL", DEFAULT_DETECTOR_URL),
                    help="URL del servidor MCP del detector")
    lb.add_argument("--local", action="store_true",
                    help="Detectar en este proceso, sin servidor (imagen GPU)")
    lb.add_argument("--limit", type=int, default=None, help="Usar solo una muestra de N imágenes")
    lb.add_argument("--seed", type=int, default=0, help="Semilla de la muestra")
    lb.add_argument("--include-unsure", action="store_true", help="Incluir también las imágenes dudosas")
    lb.add_argument("--preview", type=int, default=12, help="Imágenes en la vista previa (0 para omitirla)")

    cu = sub.add_parser("curate", help="El curador decide sobre las imágenes dudosas del filtro de calidad")
    cu.add_argument("--project", required=True)
    cu.add_argument("--limit", type=int, default=None, help="Curar solo las primeras N, para probar")

    vf = sub.add_parser("verify", help="Verifica las cajas con el modelo de visión y calcula la confianza")
    vf.add_argument("--project", required=True)
    vf.add_argument("--limit", type=int, default=None, help="Verificar solo las primeras N imágenes")
    vf.add_argument("--calibration", default=None,
                    help="Archivo de calibración (por defecto out/calibration.json)")
    vf.add_argument("--no-refs", action="store_true", help="No usar imágenes de referencia")
    vf.add_argument("--preview", type=int, default=12, help="Imágenes en la vista previa (0 para omitirla)")
    vf.add_argument("--seed", type=int, default=0)

    rn = sub.add_parser("run", help="Flujo completo: filtro, etiquetado, verificación en bucle y salidas")
    rn.add_argument("--project", required=True)
    rn.add_argument("--server", default=os.environ.get("AUTOLABEL_DETECTOR_URL", DEFAULT_DETECTOR_URL))
    rn.add_argument("--local", action="store_true",
                    help="Detectar en este proceso, sin servidor (imagen GPU)")
    rn.add_argument("--names-file", default=None,
                    help="Archivo con un nombre de imagen por línea: solo se procesan esas")
    rn.add_argument("--limit", type=int, default=None, help="Usar solo una muestra de N imágenes")
    rn.add_argument("--seed", type=int, default=0)
    rn.add_argument("--rounds", type=int, default=None, help="Máximo de estrategias a probar")
    rn.add_argument("--no-curate", action="store_true", help="No usar el curador sobre las imágenes dudosas")
    rn.add_argument("--no-refs", action="store_true", help="No usar imágenes de referencia")
    rn.add_argument("--calibration", default=None)
    rn.add_argument("--restart", action="store_true", help="Descarta el progreso guardado y empieza de nuevo")

    ls = sub.add_parser("labelstudio", help="Carga las cajas en Label Studio como preanotaciones")
    ls.add_argument("--project", required=True)
    ls.add_argument("--from-run", action="store_true",
                    help="Cargar solo las imágenes que el flujo dejó en revisión (out/run/final.json)")
    ls.add_argument("--title", default=None,
                    help="Título del proyecto en Label Studio. Por defecto el nombre del proyecto, "
                         "o ese nombre más -revision con --from-run")
    ls.add_argument("--url", default=os.environ.get("LABEL_STUDIO_URL", DEFAULT_LABEL_STUDIO_URL))
    ls.add_argument("--token", default=os.environ.get("LABEL_STUDIO_TOKEN"),
                    help="Token heredado de la API (deshabilitado por defecto en versiones recientes)")
    ls.add_argument("--username", default=os.environ.get("LABEL_STUDIO_USERNAME"))
    ls.add_argument("--password", default=os.environ.get("LABEL_STUDIO_PASSWORD"))
    ls.add_argument("--document-root", default=DEFAULT_LS_DOCUMENT_ROOT)
    ls.add_argument("--path-map", action="append", default=list(DEFAULT_PATH_MAP), metavar="DESDE=HACIA")
    ls.add_argument("--replace", action="store_true", help="Reemplaza las tareas existentes")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "init":                       # crea el proyecto, así que aún no hay nada que cargar
        run_init(args)
        return
    for noisy in ("httpx", "httpcore", "mcp"):      # una línea por petición tapa el avance real
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_project(args.project)
    if args.command == "quality":
        if args.calibrate:
            run_quality_calibration(cfg)
        else:
            run_quality(cfg, args.review)
    elif args.command == "label":
        run_label(cfg, args)
    elif args.command == "curate":
        run_curate(cfg, args)
    elif args.command == "verify":
        run_verify(cfg, args)
    elif args.command == "run":
        run_pipeline(cfg, args)
    elif args.command == "labelstudio":
        if not args.token and not (args.username and args.password):
            raise SystemExit(
                "Faltan credenciales: --username y --password (o LABEL_STUDIO_USERNAME y "
                "LABEL_STUDIO_PASSWORD), o bien un --token"
            )
        run_labelstudio(cfg, args)


if __name__ == "__main__":
    main()
