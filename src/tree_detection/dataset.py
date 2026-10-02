"""Dataset discovery and deterministic splitting helpers."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PixelWindow:
    """A rectangular image window in pixel coordinates."""

    column: int
    row: int
    width: int
    height: int


def iter_windows(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    overlap: int = 0,
) -> Iterable[PixelWindow]:
    """Yield windows that cover an image, including smaller edge windows."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if not 0 <= overlap < min(tile_width, tile_height):
        raise ValueError("overlap must be non-negative and smaller than each tile")

    step_width = tile_width - overlap
    step_height = tile_height - overlap
    for row in range(0, height, step_height):
        for column in range(0, width, step_width):
            yield PixelWindow(
                column=column,
                row=row,
                width=min(tile_width, width - column),
                height=min(tile_height, height - row),
            )


def discover_source_pairs(root: str | Path) -> list[tuple[Path, Path]]:
    """Find GeoTIFF files with a sibling GeoJSON file of the same stem."""
    root_path = Path(root)
    return [
        (raster_path, raster_path.with_suffix(".geojson"))
        for raster_path in sorted(root_path.rglob("*.tif"))
        if raster_path.with_suffix(".geojson").is_file()
    ]


def split_pairs(
    pairs: Sequence[tuple[Path, Path]],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
    list[tuple[Path, Path]],
]:
    """Split scene pairs deterministically without allowing scene leakage."""
    if train_ratio < 0 or validation_ratio < 0:
        raise ValueError("split ratios must be non-negative")
    if train_ratio + validation_ratio > 1:
        raise ValueError("train_ratio + validation_ratio cannot exceed 1")

    shuffled = sorted(pairs, key=lambda pair: str(pair[0]))
    random.Random(seed).shuffle(shuffled)
    pair_count = len(shuffled)
    train_end = round(pair_count * train_ratio)
    validation_end = train_end + round(pair_count * validation_ratio)
    return (
        shuffled[:train_end],
        shuffled[train_end:validation_end],
        shuffled[validation_end:],
    )
