"""Integration and edge-case tests for SIFT/ORB features and matching."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.features import extract_features
from src.image_io import load_images
from src.matching import match_descriptors, visualize_matches


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_feature_matching_and_visualization(method: str) -> None:
    """Run the complete feature/matching path on the first two boat images."""

    images = load_images(DEFAULT_CONFIG.data_dir)
    assert len(images) >= 2, "At least two input images are required."
    image_query, image_train = images[0], images[1]

    query_keypoints, query_descriptors = extract_features(image_query.gray, method)  # type: ignore[arg-type]
    train_keypoints, train_descriptors = extract_features(image_train.gray, method)  # type: ignore[arg-type]
    assert query_descriptors is not None
    assert train_descriptors is not None

    result = match_descriptors(query_descriptors, train_descriptors, method)  # type: ignore[arg-type]
    stats = result.statistics
    print(
        f"{method.upper()}: keypoints=({len(query_keypoints)}, {len(train_keypoints)}), "
        f"KNN candidates={stats.knn_candidates}, good matches={stats.good_matches}, "
        f"status={stats.status}"
    )

    output_path = visualize_matches(
        image_query.color,
        query_keypoints,
        image_train.color,
        train_keypoints,
        result,
        method,  # type: ignore[arg-type]
        filename="matches_01_02.jpg",
    )
    expected_dir = DEFAULT_CONFIG.output_dir / method
    assert output_path == expected_dir / "matches_01_02.jpg"
    assert output_path.is_file() and output_path.stat().st_size > 0
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert isinstance(reopened, np.ndarray) and reopened.size > 0


def test_matching_handles_missing_descriptors() -> None:
    """Missing descriptor arrays return a diagnostic empty result."""

    result = match_descriptors(None, None, "sift")
    assert result.matches == []
    assert result.statistics.status == "missing_descriptors"


def test_matching_handles_insufficient_train_descriptors() -> None:
    """KNN matching requires two train descriptors for the ratio test."""

    query = np.zeros((1, 128), dtype=np.float32)
    train = np.zeros((1, 128), dtype=np.float32)
    result = match_descriptors(query, train, "sift")
    assert result.matches == []
    assert result.statistics.status == "insufficient_train_descriptors_for_knn"


def test_feature_extractor_handles_featureless_image() -> None:
    """Uniform input returns empty keypoints and no descriptors."""

    blank = np.zeros((128, 128), dtype=np.uint8)
    for method in ("sift", "orb"):
        keypoints, descriptors = extract_features(blank, method)  # type: ignore[arg-type]
        assert keypoints == []
        assert descriptors is None


def test_visualization_rejects_non_color_input(tmp_path: Path) -> None:
    """The visualization API checks that supplied images are BGR color."""

    blank_gray = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="3-channel BGR"):
        visualize_matches(
            blank_gray,
            [],
            np.zeros((32, 32, 3), dtype=np.uint8),
            [],
            match_descriptors(None, None, "orb"),
            "orb",  # type: ignore[arg-type]
            output_dir=tmp_path,
        )

