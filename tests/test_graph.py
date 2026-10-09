import json
import textwrap
from contextlib import asynccontextmanager

import numpy as np
import pytest
from PIL import Image

from autolabel.config import load_project
from autolabel.detector_client import DetectOutput
from autolabel.detectors import Detection, Query
from autolabel.graph import Deps, run_graph
from autolabel.verification import BoxVerdict

PROJECT = """
name: demo
images: ./images
classes:
  - name: a
    prompt: "pa"
    thresholds: {auto_accept: 0.8, review: 0.4}
detector:
  agnostic_prompt: "thing"
labeling:
  max_iterations: 3
  min_gain: 0.02
  strategies:
    - {model_id: m0}
    - {model_id: m1}
    - {model_id: m2}
"""

# Qué encuentra cada estrategia en cada imagen: (caja, score)
SCRIPT = {
    "a.png": {0: [((10, 10, 60, 60), 0.9)]},                           # se acepta en la ronda 1
    "b.png": {0: [((10, 10, 60, 60), 0.6)],                             # dudosa
              1: [((10, 10, 60, 60), 0.6), ((100, 100, 150, 150), 0.95)]},   # aparece otra
    "c.png": {},                                                        # nunca hay nada
}


class FakeDetector:
    def __init__(self, strategy_index, log, fail_on_round=None):
        self.index, self.log, self.fail_on_round = strategy_index, log, fail_on_round

    async def detect(self, path):
        if self.fail_on_round == self.index:
            raise RuntimeError("falla simulada del detector")
        self.log.append((self.index, path.name))
        dets = tuple(Detection("thing", b, s) for b, s in SCRIPT.get(path.name, {}).get(self.index, []))
        return DetectOutput(320, 320, dets)


class FakeVerifier:
    """Siempre ve un objeto de la clase 'a', con la confianza igual al score del detector."""

    def __init__(self):
        self.calls = 0

    def verify_box(self, img, det, *, group_geometry=False):
        self.calls += 1
        return BoxVerdict(det.box, det.cls, det.score, 0.0, 1.0, 0.0, "a", {"a": 1.0}, group_geometry)


def make_deps(tmp_path, log, fail_on_round=None, verifier=None):
    (tmp_path / "project.yaml").write_text(textwrap.dedent(PROJECT), encoding="utf-8")
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for seed, name in enumerate(SCRIPT):
        arr = np.random.default_rng(seed).integers(0, 255, (320, 320, 3), dtype=np.uint8)
        Image.fromarray(arr).save(images / name)
    cfg = load_project(tmp_path)

    @asynccontextmanager
    async def open_detector(strategy, queries):
        index = [s.model_id for s in cfg.strategies()].index(strategy.model_id)
        yield FakeDetector(index, log, fail_on_round)

    return Deps(cfg, open_detector, verifier or FakeVerifier(), None, [Query("thing", "thing")],
                curate=False, chunk_label=2, chunk_verify=1)


def final(tmp_path):
    return json.loads((tmp_path / "out" / "run" / "final.json").read_text(encoding="utf-8"))


def test_images_follow_their_own_path_through_the_loop(tmp_path):
    log = []
    run_graph(make_deps(tmp_path, log))
    data = final(tmp_path)
    images = {i["name"]: i for i in data["images"]}

    assert images["a.png"]["status"] == "accepted" and images["a.png"]["rounds"] == 1
    assert images["b.png"]["status"] == "review" and images["b.png"]["reason"] == "cajas dudosas: 1"
    assert images["b.png"]["rounds"] == 3                         # agotó las tres estrategias
    assert images["c.png"]["status"] == "review" and images["c.png"]["reason"] == "sin cajas"
    assert data["summary"]["images"] == {"accepted": 1, "review": 2}


def test_only_unresolved_images_go_to_the_next_strategy(tmp_path):
    log = []
    run_graph(make_deps(tmp_path, log))
    by_round = {r: sorted(n for i, n in log if i == r) for r in (0, 1, 2)}
    assert by_round[0] == ["a.png", "b.png", "c.png"]
    assert by_round[1] == ["b.png", "c.png"]                       # a.png ya estaba aceptada
    assert by_round[2] == ["b.png", "c.png"]


def test_boxes_remember_which_strategies_found_them(tmp_path):
    run_graph(make_deps(tmp_path, []))
    b = next(i for i in final(tmp_path)["images"] if i["name"] == "b.png")
    first, second = sorted(b["boxes"], key=lambda x: x["box"][0])
    assert first["strategies"] == [0, 1] and first["models"] == ["m0", "m1"]       # coincidieron dos modelos
    assert second["strategies"] == [1] and second["status"] == "accepted"


def test_outputs_are_written(tmp_path):
    run_graph(make_deps(tmp_path, []))
    run = tmp_path / "out" / "run"
    assert (run / "yolo" / "labels" / "a.txt").read_text().startswith("0 ")
    assert not (run / "yolo" / "labels" / "b.txt").exists()        # solo las imágenes aceptadas
    review = json.loads((run / "review.json").read_text(encoding="utf-8"))
    assert {r["image"] for r in review} == {"b.png", "c.png"}
    assert all((run / f"round_{k}.json").exists() for k in range(3))


def test_review_list_counts_boxes_by_state_and_puts_the_doubtful_first(tmp_path):
    run_graph(make_deps(tmp_path, []))
    review = json.loads((tmp_path / "out" / "run" / "review.json").read_text(encoding="utf-8"))
    b, c = review                                                   # b tiene una caja dudosa, c ninguna caja
    assert (b["image"], b["accepted"], b["uncertain"]) == ("b.png", 1, 1)
    assert (c["image"], c["accepted"], c["uncertain"]) == ("c.png", 0, 0)
    assert (tmp_path / "out" / "run" / "review_preview.jpg").is_file()


def test_the_vlm_is_asked_once_per_box(tmp_path):
    verifier = FakeVerifier()
    run_graph(make_deps(tmp_path, [], verifier=verifier))
    assert verifier.calls == 3                                     # a: 1, b: 2, sin repetir las ya vistas


def test_a_failure_can_be_resumed_without_redoing_finished_work(tmp_path):
    log = []
    with pytest.raises(RuntimeError, match="falla simulada"):
        run_graph(make_deps(tmp_path, log, fail_on_round=1))       # la ronda 0 termina y la 1 falla
    assert [i for i, _ in log] == [0, 0, 0]

    resumed_log = []
    run_graph(make_deps(tmp_path, resumed_log))                    # misma carpeta: retoma
    assert 0 not in {i for i, _ in resumed_log}                    # no repite la ronda 0
    assert final(tmp_path)["summary"]["images"] == {"accepted": 1, "review": 2}


def test_a_finished_run_is_not_repeated_unless_asked(tmp_path):
    log = []
    deps = make_deps(tmp_path, log)
    run_graph(deps)
    calls = len(log)
    run_graph(deps)
    assert len(log) == calls
    run_graph(deps, restart=True)
    assert len(log) == 2 * calls


def test_a_restart_removes_the_round_snapshots_of_the_previous_run(tmp_path):
    deps = make_deps(tmp_path, [])
    run_graph(deps)
    run = tmp_path / "out" / "run"
    stale = run / "round_9.json"
    stale.write_text("{}", encoding="utf-8")                      # sobrante de una corrida con más rondas
    run_graph(deps, restart=True)
    assert not stale.exists()
    kept = sorted(p.name for p in run.glob("round_*.json"))
    assert kept == ["round_0.json", "round_1.json", "round_2.json"]


def test_names_restricts_the_images(tmp_path):
    log = []
    deps = make_deps(tmp_path, log)
    deps.names = ["a.png", "b.png"]
    run_graph(deps)
    assert {n for _, n in log} == {"a.png", "b.png"}
