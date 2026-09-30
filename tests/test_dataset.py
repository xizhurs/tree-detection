from pathlib import Path

import pytest

from tree_detection.dataset import PixelWindow, iter_windows, split_pairs


def test_iter_windows_covers_edges() -> None:
    assert list(iter_windows(5, 4, 3, 3, overlap=1)) == [
        PixelWindow(0, 0, 3, 3),
        PixelWindow(2, 0, 3, 3),
        PixelWindow(4, 0, 1, 3),
        PixelWindow(0, 2, 3, 2),
        PixelWindow(2, 2, 3, 2),
        PixelWindow(4, 2, 1, 2),
    ]


def test_iter_windows_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        list(iter_windows(10, 10, 4, 4, overlap=4))


def test_split_pairs_is_deterministic_and_disjoint() -> None:
    pairs = [
        (Path(f"scene-{index}.tif"), Path(f"scene-{index}.geojson"))
        for index in range(20)
    ]
    first = split_pairs(pairs)
    second = split_pairs(list(reversed(pairs)))

    assert first == second
    assert [len(split) for split in first] == [14, 3, 3]
    assert not (set(first[0]) & set(first[1]) | set(first[0]) & set(first[2]))
