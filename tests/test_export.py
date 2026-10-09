import pytest

from autolabel.detectors import Detection
from autolabel.export import write_yolo, yolo_lines
from autolabel.labeling import ImageLabels


def item(*dets, width=200, height=100):
    return ImageLabels("img.png", width, height, tuple(dets))


def test_yolo_line_is_normalized_center_and_size():
    line = yolo_lines(item(Detection("x", (20, 10, 60, 50), 0.9)), {"x": 3})[0]
    cls, cx, cy, w, h = line.split()
    assert cls == "3"
    assert [float(v) for v in (cx, cy, w, h)] == pytest.approx([0.2, 0.3, 0.2, 0.4])


def test_boxes_are_clipped_to_the_image():
    line = yolo_lines(item(Detection("x", (-10, 80, 120, 150), 0.9)), {"x": 0})[0]
    _, cx, cy, w, h = (float(v) for v in line.split())
    assert (cx, cy, w, h) == pytest.approx((0.3, 0.9, 0.6, 0.2))
    assert all(0 <= v <= 1 for v in (cx, cy, w, h))


def test_degenerate_boxes_are_skipped():
    no_width = Detection("x", (50, 50, 50, 80), 0.9)
    outside = Detection("x", (300, 10, 320, 30), 0.9)
    dets = [no_width, outside]
    assert yolo_lines(item(*dets), {"x": 0}) == []


def test_write_yolo_files(tmp_path):
    results = [
        ImageLabels("a.png", 100, 100, (Detection("x", (10, 10, 50, 50), 0.9),)),
        ImageLabels("b.png", 100, 100, ()),
    ]
    out = write_yolo(results, ["x", "y"], tmp_path / "imgs", tmp_path / "yolo")
    assert (out / "labels" / "a.txt").read_text().startswith("0 0.300000 0.300000 0.400000 0.400000")
    assert (out / "labels" / "b.txt").read_text() == ""          # vacío = sin objetos
    assert (out / "classes.txt").read_text() == "x\ny\n"
    assert (out / "images.txt").read_text().splitlines() == [
        str(tmp_path / "imgs" / "a.png"), str(tmp_path / "imgs" / "b.png"),
    ]
    data = (out / "data.yaml").read_text()
    assert "0: x" in data and "1: y" in data and "train: images.txt" in data
