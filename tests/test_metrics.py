import pytest

from eval.metrics import build_curve, match_image

GT = [[0, 0, 10, 10], [20, 20, 30, 30]]


def curve(images, iou=0.5):
    return build_curve(images, iou)


def test_perfect_detection():
    c = curve([(GT, [0.9, 0.8], GT, [])])
    assert c.ap() == pytest.approx(1.0)
    best = c.best_f1()
    assert best["f1"] == pytest.approx(1.0)
    assert best["threshold"] == pytest.approx(0.8)


def test_false_positive_lowers_precision_but_not_recall():
    dets = [*GT, [50, 50, 60, 60]]
    c = curve([(dets, [0.9, 0.8, 0.7], GT, [])])
    at_low = c.at(0.5)
    assert at_low["recall"] == pytest.approx(1.0)
    assert at_low["precision"] == pytest.approx(2 / 3)
    assert c.at(0.75)["precision"] == pytest.approx(1.0)


def test_missed_object_lowers_recall():
    c = curve([([GT[0]], [0.9], GT, [])])
    assert c.at(0.5)["recall"] == pytest.approx(0.5)
    assert c.ap() == pytest.approx(0.5)


def test_duplicate_detection_counts_as_false_positive():
    scores, tp = match_image([GT[0], GT[0]], [0.9, 0.8], [GT[0]], [], 0.5)
    assert tp.tolist() == [True, False]
    assert scores.tolist() == [0.9, 0.8]


def test_detection_over_ignored_box_is_not_penalized():
    small = [[100, 100, 104, 104]]
    scores, tp = match_image(small, [0.9], [], small, 0.5)
    assert len(scores) == 0 and len(tp) == 0


def test_no_ground_truth_and_no_detections():
    c = curve([([], [], [], [])])
    assert c.ap() == 0.0
    assert c.best_f1()["f1"] == 0.0


def test_scores_are_pooled_across_images():
    c = curve([(GT[:1], [0.9], GT[:1], []), ([[50, 50, 60, 60]], [0.95], GT[1:], [])])
    # La caja falsa de la segunda imagen tiene el mayor score
    assert c.scores[0] == pytest.approx(0.95)
    assert c.n_gt == 2
    assert c.at(0.5) == pytest.approx(c.at(0.5))  # estable
    assert c.at(0.5)["tp"] == 1 and c.at(0.5)["fp"] == 1
