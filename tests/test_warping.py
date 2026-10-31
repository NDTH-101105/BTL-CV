"""Unit and image-pair integration tests for perspective warping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.blending import blend_images, save_blend_result
from src.features import extract_features
from src.homography import estimate_homography
from src.image_io import load_images
from src.matching import match_descriptors
from src.warping import WarpingError, save_warped_images, warp_images_to_reference


def test_identity_warp_preserves_single_image() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[2:10, 3:14] = (20, 100, 220)
    result = warp_images_to_reference([image], [], reference_index=0)
    assert result.canvas_size == (16, 12)
    assert np.array_equal(result.images[0].image, image)
    assert np.all(result.images[0].mask == 255)
    assert np.allclose(result.images[0].source_to_canvas, np.eye(3))


def test_negative_coordinates_expand_canvas_without_clipping() -> None:
    first = np.full((8, 6, 3), (0, 0, 220), dtype=np.uint8)
    second = np.full((8, 6, 3), (220, 0, 0), dtype=np.uint8)
    # H_0 maps image 0 to image 1 by -3px. With image 1 as reference,
    # image 0 therefore occupies x=-3..2 and requires a +3px canvas shift.
    pairwise = [np.array([[1.0, 0.0, -3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]
    result = warp_images_to_reference([first, second], pairwise, reference_index=1)
    assert result.bounds == (-3, 0, 5, 7)
    assert result.canvas_size == (9, 8)
    assert np.all(result.images[0].mask[:, :6] == 255)
    assert np.all(result.images[1].mask[:, 3:] == 255)
    assert np.all(result.images[0].image[:, :6] == first)
    assert np.all(result.images[1].image[:, 3:] == second)


def test_warp_rejects_singular_homography() -> None:
    image = np.ones((4, 4, 3), dtype=np.uint8)
    singular = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="zero matrix|singular"):
        warp_images_to_reference([image, image], [singular])


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_warp_estimated_boat_pair(method: str, tmp_path: Path) -> None:
    images = load_images(DEFAULT_CONFIG.data_dir)
    source, destination = images[0], images[1]
    source_keypoints, source_descriptors = extract_features(source.gray, method)  # type: ignore[arg-type]
    destination_keypoints, destination_descriptors = extract_features(destination.gray, method)  # type: ignore[arg-type]
    assert source_descriptors is not None and destination_descriptors is not None
    good = match_descriptors(source_descriptors, destination_descriptors, method).matches  # type: ignore[arg-type]
    homography = estimate_homography(source_keypoints, destination_keypoints, good)

    warped = warp_images_to_reference(
        [source.color, destination.color], [homography.H], reference_index=0
    )
    assert len(warped.images) == 2
    assert warped.images[0].image.shape == warped.images[1].image.shape
    assert warped.images[0].mask.shape == warped.images[1].mask.shape
    assert np.any(warped.images[0].mask)
    assert np.any(warped.images[1].mask)
    assert np.any((warped.images[0].mask > 0) & (warped.images[1].mask > 0))

    output_dir = tmp_path / method
    paths = save_warped_images(warped, output_dir)
    assert len(paths) == 4
    assert all(path.is_file() and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None for path in paths)
    blended = blend_images(
        [item.image for item in warped.images],
        [item.mask for item in warped.images],
    )
    panorama_paths = save_blend_result(blended, output_dir)
    assert len(panorama_paths) == 4
    assert all(
        path.is_file() and cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
        for path in panorama_paths
    )

