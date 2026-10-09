"""Mosaico: partir una imagen en tiles solapados para detectar objetos chicos.

Un detector reduce la imagen a un tamaño fijo, así que los objetos pequeños se pierden. Al
detectar por tiles cada objeto se ve más grande. Solo geometría, sin modelos.
"""
from __future__ import annotations

import math

Tile = tuple[int, int, int, int]


def tile_grid(width: int, height: int, tiles: int, overlap: float = 0.2) -> list[Tile]:
    """Grilla de ``tiles x tiles`` regiones que se extienden ``overlap`` más allá de su celda.

    El solape evita que un objeto sobre la unión quede partido en todos los tiles.
    """
    if tiles < 1:
        raise ValueError("tiles debe ser al menos 1")
    cell_w, cell_h = width / tiles, height / tiles
    grow_w, grow_h = cell_w * overlap / 2, cell_h * overlap / 2
    grid: list[Tile] = []
    for row in range(tiles):
        for col in range(tiles):
            grid.append((
                int(max(0, col * cell_w - grow_w)),
                int(max(0, row * cell_h - grow_h)),
                min(width, math.ceil((col + 1) * cell_w + grow_w)),
                min(height, math.ceil((row + 1) * cell_h + grow_h)),
            ))
    return grid


def is_cut_by_tile(
    box: tuple[float, float, float, float],
    tile: Tile,
    image_size: tuple[int, int],
    margin: float = 3.0,
) -> bool:
    """Una caja que toca el borde del tile, y ese borde no es el de la imagen, está probablemente partida.

    ``box`` está en coordenadas de la imagen completa.
    """
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    width, height = image_size
    return (
        (tx1 > 0 and x1 <= tx1 + margin)
        or (ty1 > 0 and y1 <= ty1 + margin)
        or (tx2 < width and x2 >= tx2 - margin)
        or (ty2 < height and y2 >= ty2 - margin)
    )
