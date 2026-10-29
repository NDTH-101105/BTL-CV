"""Descriptor matching and visualization for SIFT and ORB features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DEFAULT_CONFIG, MatchingConfig
from src.features import FeatureMethod, _normalize_method


@dataclass(frozen=True)
class MatchStatistics:
    """Counts and status explaining the output of a descriptor match."""

    query_descriptors: int
    train_descriptors: int
    knn_candidates: int
    good_matches: int
    minimum_required: int
    status: str

    @property
    def has_enough_matches(self) -> bool:
        """Whether the good match count meets the configured minimum."""

        return self.good_matches >= self.minimum_required


@dataclass(frozen=True)
class MatchResult:
    """Ratio-filtered matches and their summary statistics."""

    matches: list[Any]
    statistics: MatchStatistics


def match_descriptors(
    descriptors_query: NDArray[Any] | None,
    descriptors_train: NDArray[Any] | None,
    method: FeatureMethod,
    config: MatchingConfig | None = None,
) -> MatchResult:
    """Match descriptors using BFMatcher, KNN (k=2), and Lowe's ratio test.

    Missing or too-short descriptor sets return an empty result with a status
    instead of raising. Malformed arrays and invalid configuration raise errors.
    """

    normalized = _normalize_method(method)
    selected_config = DEFAULT_CONFIG.matching if config is None else config
    if not isinstance(selected_config, MatchingConfig):
        raise TypeError("config must be a MatchingConfig instance.")
    if not 0.0 < selected_config.ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be greater than 0 and less than 1.")
    if selected_config.minimum_matches < 0:
        raise ValueError("minimum_matches must not be negative.")

    query_count = 0 if descriptors_query is None else len(descriptors_query)
    train_count = 0 if descriptors_train is None else len(descriptors_train)

    def empty_result(candidate_count: int, status: str) -> MatchResult:
        return MatchResult(
            matches=[],
            statistics=MatchStatistics(
                query_descriptors=query_count,
                train_descriptors=train_count,
                knn_candidates=candidate_count,
                good_matches=0,
                minimum_required=selected_config.minimum_matches,
                status=status,
            ),
        )

    if descriptors_query is None or descriptors_train is None:
        return empty_result(0, "missing_descriptors")
    for name, descriptors in (
        ("descriptors_query", descriptors_query),
        ("descriptors_train", descriptors_train),
    ):
        if not isinstance(descriptors, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array or None.")
        if descriptors.ndim != 2:
            raise ValueError(f"{name} must be a 2-D array; got shape {descriptors.shape}.")
        if descriptors.shape[1] == 0:
            raise ValueError(f"{name} must have at least one descriptor element.")

    if query_count == 0:
        return empty_result(0, "empty_query_descriptors")
    if train_count < 2:
        return empty_result(0, "insufficient_train_descriptors_for_knn")

    if normalized == "sift":
        if descriptors_query.dtype not in (np.dtype(np.float32), np.dtype(np.uint8)):
            raise TypeError("SIFT descriptors must have dtype float32 (or supported uint8).")
        if descriptors_train.dtype != descriptors_query.dtype:
            raise TypeError("Query and train SIFT descriptors must have the same dtype.")
        norm_type = getattr(cv2, selected_config.norm_name_sift, None)
        if norm_type is None:
            raise ValueError(f"Unknown OpenCV norm: {selected_config.norm_name_sift}")
    else:
        if descriptors_query.dtype != np.uint8 or descriptors_train.dtype != np.uint8:
            raise TypeError("ORB descriptors must have dtype uint8.")
        norm_type = getattr(cv2, selected_config.norm_name_orb, None)
        if norm_type is None:
            raise ValueError(f"Unknown OpenCV norm: {selected_config.norm_name_orb}")
        if descriptors_query.shape[1] != descriptors_train.shape[1]:
            raise ValueError("Query and train descriptors must have the same width.")

    matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_query, descriptors_train, k=2)
    good_matches = [
        pair[0]
        for pair in knn_matches
        if len(pair) >= 2 and pair[0].distance < selected_config.ratio_threshold * pair[1].distance
    ]
    status = "ok" if len(good_matches) >= selected_config.minimum_matches else "insufficient_good_matches"
    if not good_matches:
        status = "no_good_matches"

    return MatchResult(
        matches=good_matches,
        statistics=MatchStatistics(
            query_descriptors=query_count,
            train_descriptors=train_count,
            knn_candidates=sum(1 for pair in knn_matches if len(pair) >= 2),
            good_matches=len(good_matches),
            minimum_required=selected_config.minimum_matches,
            status=status,
        ),
    )


def visualize_matches(
    image_query: NDArray[np.uint8],
    keypoints_query: list[Any],
    image_train: NDArray[np.uint8],
    keypoints_train: list[Any],
    result: MatchResult,
    method: FeatureMethod,
    *,
    output_dir: str | Path | None = None,
    filename: str = "matches_01_02.jpg",
) -> Path:
    """Draw and save ratio-filtered matches; return the saved image path.

    The default output location is ``outputs/sift`` or ``outputs/orb``.
    Inputs are expected to be BGR color images, such as ``LoadedImage.color``.
    """

    normalized = _normalize_method(method)
    for name, image in (("image_query", image_query), ("image_train", image_train)):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"{name} must be a non-empty 3-channel BGR image.")
        if image.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8.")
    if not isinstance(result, MatchResult):
        raise TypeError("result must be a MatchResult instance.")
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a non-empty file name without directory parts.")

    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_CONFIG.output_dir / normalized
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    visualization = cv2.drawMatches(
        image_query,
        keypoints_query,
        image_train,
        keypoints_train,
        result.matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if visualization is None or visualization.size == 0:
        raise RuntimeError("OpenCV did not produce a match visualization.")
    if not cv2.imwrite(str(output_path), visualization):
        raise OSError(f"Could not save match visualization: {output_path}")
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if reopened is None or reopened.size == 0:
        raise OSError(f"Saved visualization could not be reopened: {output_path}")
    return output_path

