import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from autolabel.config import QualityConfig
from autolabel.quality import (
    QualityMetrics,
    assess_batch,
    find_duplicate_pairs,
    judge,
    measure_image,
    sharpness,
)

ROOT = Path(__file__).resolve().parent.parent


def noise_gray(seed: int, size=(500, 400)) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size, dtype=np.uint8)


def save_gray(path, gray: np.ndarray) -> str:
    Image.fromarray(gray).convert("RGB").save(path)
    return str(path)


def metrics(**over) -> QualityMetrics:
    base = {"width": 400, "height": 500, "blur": 500.0, "brightness": 120.0, "clipped": 0.0, "phash": 0}
    return QualityMetrics(**(base | over))


# --------------------------------------------------------------------------- nitidez


def test_sharpness_drops_monotonically_with_blur():
    sharp = noise_gray(0)
    values = [sharpness(sharp)] + [
        sharpness(cv2.GaussianBlur(sharp, (0, 0), s)) for s in (1, 2, 4)
    ]
    assert values == sorted(values, reverse=True)
    assert values[0] > 10 * values[-1]


def test_uniform_image_has_zero_sharpness():
    assert sharpness(np.full((300, 300), 128, np.uint8)) == pytest.approx(0.0)


def test_sharpness_is_not_penalized_by_large_smooth_areas():
    # Mitad lisa y mitad con textura: tiene bordes nítidos, no debe contar como borrosa
    img = np.full((400, 400), 128, np.uint8)
    img[:, 200:] = noise_gray(1, (400, 200))
    assert sharpness(img) > 0.5 * sharpness(noise_gray(1, (400, 400)))


# --------------------------------------------------------------------------- veredicto


@pytest.mark.parametrize(
    ("changes", "verdict", "reason"),
    [
        ({}, "keep", None),
        ({"blur": 30.0}, "unsure", "slightly_blurry"),
        ({"blur": 10.0}, "drop", "blurry"),
        ({"width": 200}, "drop", "low_resolution"),
        ({"brightness": 5.0}, "drop", "too_dark"),
        ({"brightness": 250.0}, "drop", "too_bright"),
        ({"clipped": 0.6}, "drop", "clipped"),
    ],
)
def test_judge(changes, verdict, reason):
    got_verdict, reasons = judge(metrics(**changes), QualityConfig(blur_min=25, blur_ok=35))
    assert got_verdict == verdict
    assert (reason in reasons) if reason else reasons == ()


def test_without_blur_cutoffs_blur_is_not_judged():
    cfg = QualityConfig()                                   # sin cortes de nitidez
    assert judge(metrics(blur=0.5), cfg) == ("keep", ())
    assert judge(metrics(brightness=1.0), cfg)[0] == "drop"  # lo demás sí se juzga


def test_blur_cutoffs_come_in_pairs():
    with pytest.raises(ValidationError, match="juntos"):
        QualityConfig(blur_min=10)
    with pytest.raises(ValidationError, match="juntos"):
        QualityConfig(blur_ok=10)


def test_calibration_report_gives_decreasing_anchors_and_usable_cutoffs(tmp_path):
    from autolabel.quality import calibration_report

    paths = [save_gray(tmp_path / f"{i}.png", cv2.GaussianBlur(noise_gray(i), (0, 0), 0.3 + 0.5 * i))
             for i in range(10)]
    report = calibration_report(paths)
    a = report["anchors"]
    assert a[1.0] > a[2.0] > a[3.0] > 0
    assert report["suggested"] == {"blur_min": a[2.0], "blur_ok": a[1.0]}
    QualityConfig(**report["suggested"])                    # los cortes sugeridos son válidos
    assert report["sorted_blur"] == sorted(report["sorted_blur"])
    assert len(report["sorted_paths"]) == report["count"] == 10


def test_calibration_report_skips_unreadable_images(tmp_path):
    from autolabel.quality import calibration_report

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"no es una imagen")
    good = save_gray(tmp_path / "g.png", noise_gray(1))
    assert calibration_report([str(bad), good])["count"] == 1


def test_cutoffs_are_inclusive_at_the_boundaries():
    cfg = QualityConfig(blur_min=25, blur_ok=35)
    assert judge(metrics(blur=25.0), cfg)[0] == "unsure"
    assert judge(metrics(blur=35.0), cfg)[0] == "keep"


def test_quality_config_validation():
    with pytest.raises(ValidationError):
        QualityConfig(blur_min=50, blur_ok=10)
    with pytest.raises(ValidationError):
        QualityConfig(brightness_range=(200, 100))
    with pytest.raises(ValidationError):
        QualityConfig(unknown_key=1)


# --------------------------------------------------------------------------- duplicados


def test_find_duplicate_pairs():
    a, b = 0b1010, 0b1011                         # distancia 1
    far = (1 << 64) - 1                           # muy distinto
    assert find_duplicate_pairs([a, b, far], max_distance=2) == [(0, 1)]
    assert find_duplicate_pairs([a, b, far], max_distance=0) == []
    assert find_duplicate_pairs([a], max_distance=4) == []
    assert find_duplicate_pairs([], max_distance=4) == []


def test_perceptual_hash_is_stable_to_small_changes(tmp_path):
    base = noise_gray(2)
    p1 = save_gray(tmp_path / "a.png", base)
    p2 = save_gray(tmp_path / "b.png", np.clip(base.astype(int) + 8, 0, 255).astype(np.uint8))
    p3 = save_gray(tmp_path / "c.png", noise_gray(3))
    h1, h2, h3 = (measure_image(p).phash for p in (p1, p2, p3))
    assert (h1 ^ h2).bit_count() <= 4
    assert (h1 ^ h3).bit_count() > 15


# --------------------------------------------------------------------------- lote


def test_assess_batch_keeps_sharpest_duplicate_and_preserves_order(tmp_path):
    sharp = noise_gray(4)
    soft = cv2.GaussianBlur(sharp, (0, 0), 1.2)
    paths = [
        save_gray(tmp_path / "1_soft_copy.png", soft),
        save_gray(tmp_path / "2_other.png", noise_gray(5)),
        save_gray(tmp_path / "3_sharp_original.png", sharp),
    ]
    cfg = QualityConfig(blur_min=1, blur_ok=2)
    result = assess_batch(paths, cfg)
    assert [Path(a.image).name for a in result] == [Path(p).name for p in paths]
    assert result[0].verdict == "drop"
    assert result[0].reasons == ("duplicate_of:3_sharp_original.png",)
    assert result[1].verdict == "keep"
    assert result[2].verdict == "keep"


def test_dedupe_can_be_disabled(tmp_path):
    img = noise_gray(6)
    paths = [save_gray(tmp_path / "a.png", img), save_gray(tmp_path / "b.png", img)]
    result = assess_batch(paths, QualityConfig(dedupe=False, blur_min=1, blur_ok=2))
    assert [a.verdict for a in result] == ["keep", "keep"]


def test_unreadable_file_is_dropped_without_breaking_the_batch(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"esto no es una imagen")
    good = save_gray(tmp_path / "good.png", noise_gray(7))
    result = assess_batch([str(bad), good], QualityConfig(blur_min=1, blur_ok=2))
    assert result[0].verdict == "drop" and result[0].reasons == ("unreadable",)
    assert result[0].metrics is None
    assert result[1].verdict == "keep"


def test_empty_batch():
    assert assess_batch([], QualityConfig()) == []


def test_dark_and_small_images_are_dropped(tmp_path):
    dark = save_gray(tmp_path / "dark.png", np.full((500, 400), 8, np.uint8))
    small = save_gray(tmp_path / "small.png", noise_gray(8, (100, 80)))
    result = assess_batch([dark, small], QualityConfig(dedupe=False))
    assert result[0].verdict == "drop" and "too_dark" in result[0].reasons
    assert result[1].verdict == "drop" and "low_resolution" in result[1].reasons


# --------------------------------------------------------------------------- servidor MCP


def test_mcp_server_end_to_end(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    sharp = save_gray(tmp_path / "sharp.png", noise_gray(9))
    blurry = save_gray(tmp_path / "blurry.png", cv2.GaussianBlur(noise_gray(10), (0, 0), 8))

    async def call():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "services.mcp_quality"],
            cwd=str(ROOT),
            env={"AUTOLABEL_ALLOWED_ROOTS": str(tmp_path)},
        )
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            ok = await session.call_tool(
                "assess_images",
                {"image_paths": [sharp, blurry], "params": {"blur_min": 25, "blur_ok": 35}},
            )
            outside = await session.call_tool("assess_images", {"image_paths": ["/etc/passwd"]})
            return ok, outside

    ok, outside = asyncio.run(call())
    assert not ok.isError
    verdicts = {Path(r["image"]).name: r["verdict"] for r in ok.structuredContent["result"]}
    assert verdicts == {"sharp.png": "keep", "blurry.png": "drop"}
    assert outside.isError          # rutas fuera de las carpetas permitidas se rechazan
