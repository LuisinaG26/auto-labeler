import numpy as np
import pytest

from autolabel.geometry import (
    as_boxes,
    box_sizes,
    clip_boxes,
    iou_matrix,
    polygon_to_box,
    scale_boxes,
)


def test_polygon_to_box():
    assert polygon_to_box([(10, 5), (30, 8), (20, 40)]) == [10.0, 5.0, 30.0, 40.0]


def test_iou_identical_disjoint_and_half():
    a = [[0, 0, 10, 10]]
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0.0
    # Mitad solapada: inter 50, unión 150
    assert iou_matrix(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)


def test_iou_matrix_shape_and_empty_inputs():
    m = iou_matrix([[0, 0, 4, 4], [1, 1, 5, 5]], [[0, 0, 4, 4], [2, 2, 6, 6], [9, 9, 10, 10]])
    assert m.shape == (2, 3)
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_degenerate_boxes_do_not_divide_by_zero():
    assert iou_matrix([[1, 1, 1, 1]], [[1, 1, 1, 1]])[0, 0] == 0.0


def test_as_boxes_accepts_empty_list():
    assert as_boxes([]).shape == (0, 4)


def test_scale_clip_and_sizes():
    scaled = scale_boxes([[10, 20, 30, 60]], 0.5, 0.25)
    assert scaled.tolist() == [[5.0, 5.0, 15.0, 15.0]]
    clipped = clip_boxes([[-5, -5, 120, 90]], 100, 80)
    assert clipped.tolist() == [[0.0, 0.0, 100.0, 80.0]]
    assert np.allclose(box_sizes([[0, 0, 4, 6]]), [[4, 6]])
