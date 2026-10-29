"""Tests for RANSAC homography estimation and inlier visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.features import extract_features
from src.homography import (
    HomographyEstimationError,
    HomographyResult,
    estimate_homography,
    visualize_homography_matches,
)
from src.image_io import load_images
from src.matching import match_descriptors


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_homography_on_boat_pair(method: str) -> None:
    """Estimate and visualize a transform for the first adjacent boat pair."""

    images = load_images(DEFAULT_CONFIG.data_dir)
    assert len(images) >= 2, "At least two input images are required."
    source, destination = images[0], images[1]
    source_keypoints, source_descriptors = extract_features(source.gray, method)  # type: ignore[arg-type]
    destination_keypoints, destination_descriptors = extract_features(destination.gray, method)  # type: ignore[arg-type]
    assert source_descriptors is not None
    assert destination_descriptors is not None

    match_result = match_descriptors(source_descriptors, destination_descriptors, method)  # type: ignore[arg-type]
    result = estimate_homography(source_keypoints, destination_keypoints, match_result.matches)

    assert result.H.shape == (3, 3)
    assert np.isfinite(result.H).all()
    assert np.linalg.matrix_rank(result.H) == 3
    assert result.inlier_mask.dtype == np.bool_
    assert result.inlier_count == int(result.inlier_mask.sum())
    assert result.inlier_count >= DEFAULT_CONFIG.ransac.minimum_inliers
    assert result.inlier_ratio == pytest.approx(result.inlier_count / result.total_matches)
    assert np.isfinite(result.reprojection_error)
    assert result.reprojection_error >= 0.0

    print(
        f"{method.upper()}: good_matches={result.total_matches}, "
        f"inliers={result.inlier_count}, inlier_ratio={result.inlier_ratio:.4f}, "
        f"mean_inlier_reprojection_error={result.reprojection_error:.4f}px"
    )
    output_path = visualize_homography_matches(
        source.color,
        source_keypoints,
        destination.color,
        destination_keypoints,
        match_result.matches,
        result,
        method,  # type: ignore[arg-type]
        filename="homography_01_02.jpg",
    )
    assert output_path == DEFAULT_CONFIG.output_dir / method / "homography_01_02.jpg"
    assert output_path.is_file() and output_path.stat().st_size > 0
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert reopened is not None and reopened.size > 0


def test_homography_rejects_too_few_matches() -> None:
    """Four correspondences are the minimum for estimating a homography."""

    keypoints = [cv2.KeyPoint(float(i), float(i), 1.0) for i in range(3)]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(3)]
    with pytest.raises(HomographyEstimationError, match="At least 4 point matches"):
        estimate_homography(keypoints, keypoints, matches)


def test_homography_reports_cv_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed OpenCV estimate is surfaced as a clear domain error."""

    keypoints = [cv2.KeyPoint(float(i % 2), float(i // 2), 1.0) for i in range(4)]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(4)]

    def fail_estimation(*args: Any, **kwargs: Any) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr("src.homography.cv2.findHomography", fail_estimation)
    with pytest.raises(HomographyEstimationError, match="returned no homography"):
        estimate_homography(keypoints, keypoints, matches)


def test_homography_rejects_singular_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-invertible 3x3 result is rejected as a degenerate transform."""

    keypoints = [cv2.KeyPoint(float(i % 2), float(i // 2), 1.0) for i in range(4)]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(4)]
    singular = np.zeros((3, 3), dtype=np.float64)
    mask = np.ones((4, 1), dtype=np.uint8)

    def singular_estimation(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return singular, mask

    monkeypatch.setattr("src.homography.cv2.findHomography", singular_estimation)
    with pytest.raises(HomographyEstimationError, match="zero matrix|singular"):
        estimate_homography(keypoints, keypoints, matches)

