import pytest

from autolabel.tiling import is_cut_by_tile, tile_grid


def test_single_tile_is_the_whole_image():
    assert tile_grid(200, 100, 1) == [(0, 0, 200, 100)]


def test_grid_covers_the_image_and_tiles_overlap():
    grid = tile_grid(400, 400, 2, overlap=0.2)
    assert len(grid) == 4
    xs = {t[0] for t in grid} | {t[2] for t in grid}
    assert min(xs) == 0 and max(xs) == 400
    first, second = grid[0], grid[1]                       # vecinos en la misma fila
    assert second[0] < first[2]                            # se solapan
    assert all(0 <= x1 < x2 <= 400 and 0 <= y1 < y2 <= 400 for x1, y1, x2, y2 in grid)


def test_every_point_belongs_to_some_tile_with_room_around_it():
    grid = tile_grid(300, 200, 3, overlap=0.3)
    for x in range(0, 300, 7):
        for y in range(0, 200, 7):
            assert any(t[0] <= x < t[2] and t[1] <= y < t[3] for t in grid)


def test_invalid_tile_count():
    with pytest.raises(ValueError):
        tile_grid(100, 100, 0)


def test_box_touching_an_inner_tile_edge_is_cut_but_not_one_on_the_image_border():
    tile = (0, 0, 220, 220)                                # tile arriba a la izquierda de una imagen de 400
    size = (400, 400)
    assert is_cut_by_tile((200, 50, 221, 90), tile, size)       # toca el borde derecho interior
    assert not is_cut_by_tile((0, 50, 40, 90), tile, size)      # borde izquierdo: es el de la imagen
    assert not is_cut_by_tile((50, 50, 90, 90), tile, size)     # adentro


def test_tile_far_from_the_origin_cuts_on_every_inner_side():
    tile = (180, 180, 400, 400)                            # tile abajo a la derecha
    size = (400, 400)
    assert is_cut_by_tile((181, 250, 220, 280), tile, size)     # borde izquierdo interior
    assert is_cut_by_tile((250, 182, 280, 220), tile, size)     # borde superior interior
    assert not is_cut_by_tile((300, 300, 399, 399), tile, size)  # esos bordes son los de la imagen
