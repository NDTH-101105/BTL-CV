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


def test_roi_feather_matches_full_canvas_reference_within_one_level() -> None:
    """The ROI/float32 optimization preserves the former feather weights."""

    rng = np.random.default_rng(41)
    first = rng.integers(0, 256, (48, 72, 3), dtype=np.uint8)
    second = rng.integers(0, 256, (48, 72, 3), dtype=np.uint8)
    first_mask = np.zeros((48, 72), dtype=np.uint8)
    second_mask = np.zeros((48, 72), dtype=np.uint8)
    first_mask[3:45, 2:55] = 255
    second_mask[6:46, 24:70] = 255

    actual = blend_images([first, second], [first_mask, second_mask]).panorama

    valid_masks = [first_mask > 0, second_mask > 0]
    coverage = valid_masks[0].astype(np.uint16) + valid_masks[1].astype(np.uint16)
    overlap = coverage >= 2
    expected = np.zeros_like(first)
    exclusive_first = valid_masks[0] & (coverage == 1)
    exclusive_second = valid_masks[1] & (coverage == 1)
    expected[exclusive_first] = first[exclusive_first]
    expected[exclusive_second] = second[exclusive_second]
    accumulated = np.zeros(first.shape, dtype=np.float64)
    total_weight = np.zeros(first.shape[:2], dtype=np.float64)
    for image, valid in zip((first, second), valid_masks, strict=True):
        distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
        distance = np.maximum(distance, 1.0) * valid
        accumulated += image.astype(np.float64) * distance[:, :, None]
        total_weight += distance
    expected[overlap] = np.clip(
        np.rint(accumulated[overlap] / total_weight[overlap, None]), 0, 255
    ).astype(np.uint8)

    difference = cv2.absdiff(actual, expected)
    assert int(difference.max()) <= 1


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
    assert len(paths) == 4
    assert all(path.is_file() and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None for path in paths)

    rgba = cv2.imread(str(tmp_path / "panorama_transparent.png"), cv2.IMREAD_UNCHANGED)
    assert rgba is not None and rgba.shape == (5, 4, 4)
    assert np.array_equal(rgba[:, :, 3], result.mask)
    assert np.array_equal(rgba[:, :, :3], result.panorama)

