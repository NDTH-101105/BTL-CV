"""Tests for mask-aware feather blending."""

from __future__ import annotations

import cv2
import numpy as np

from src.blending import blend_images, save_blend_result


def test_blends_overlap_and_preserves_exclusive_pixels() -> None:
    first = np.full((4, 6, 3), 40, dtype=np.uint8)
    second = np.full((4, 6, 3), 200, dtype=np.uint8)
    first_mask = np.zeros((4, 6), dtype=np.uint8)
    second_mask = np.zeros((4, 6), dtype=np.uint8)
    first_mask[:, :4] = 255
    second_mask[:, 2:] = 255

    result = blend_images([first, second], [first_mask, second_mask])
    assert np.all(result.panorama[:, :2] == 40)
    assert np.all(result.panorama[:, 4:] == 200)
    assert np.all((result.panorama[:, 2:4] >= 40) & (result.panorama[:, 2:4] <= 200))
    assert np.all(result.overlap_mask[:, :2] == 0)
    assert np.all(result.overlap_mask[:, 2:4] == 255)
    assert np.all(result.mask == 255)


def test_empty_masks_produce_empty_black_result() -> None:
    image = np.full((3, 5, 3), 123, dtype=np.uint8)
    empty = np.zeros((3, 5), dtype=np.uint8)
    result = blend_images([image], [empty])
    assert not np.any(result.panorama)
    assert not np.any(result.mask)
    assert not np.any(result.overlap_mask)


def test_no_overlap_keeps_each_image_region_exactly() -> None:
    first = np.full((3, 4, 3), 31, dtype=np.uint8)
    second = np.full((3, 4, 3), 209, dtype=np.uint8)
    first_mask = np.zeros((3, 4), dtype=np.uint8)
    second_mask = np.zeros((3, 4), dtype=np.uint8)
    first_mask[:, :2] = 255
    second_mask[:, 2:] = 255
    result = blend_images([first, second], [first_mask, second_mask])
    assert not np.any(result.overlap_mask)
    assert np.all(result.panorama[:, :2] == 31)
    assert np.all(result.panorama[:, 2:] == 209)


def test_different_image_sizes_are_padded_and_saved(tmp_path) -> None:
    first = np.full((3, 4, 3), (0, 30, 200), dtype=np.uint8)
    second = np.full((5, 2, 3), (200, 30, 0), dtype=np.uint8)
    first_mask = np.full((3, 4), 255, dtype=np.uint8)
    second_mask = np.full((5, 2), 255, dtype=np.uint8)
    result = blend_images([first, second], [first_mask, second_mask])
    assert result.panorama.shape == (5, 4, 3)
    assert result.mask.shape == (5, 4)
    assert not np.any(result.mask[3:, 2:])

    paths = save_blend_result(result, tmp_path)
    assert len(paths) == 3
    assert all(path.is_file() and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None for path in paths)

