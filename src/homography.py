"""Robust homography estimation and correspondence visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DEFAULT_CONFIG, RansacConfig
from src.features import FeatureMethod, _normalize_method


class HomographyEstimationError(RuntimeError):
    """Raised when correspondences cannot produce a valid homography."""


@dataclass(frozen=True)
class HomographyResult:
    """Homography mapping source-image coordinates into destination coordinates."""

    H: NDArray[np.float64]
    inlier_mask: NDArray[np.bool_]
    inlier_count: int
    inlier_ratio: float
    reprojection_error: float
    total_matches: int


def estimate_homography(
    keypoints_source: list[Any],
    keypoints_destination: list[Any],
    matches: list[Any],
    config: RansacConfig | None = None,
) -> HomographyResult:
    """Estimate the transform mapping source points into destination points.

    OpenCV ``DMatch.queryIdx`` indexes ``keypoints_source`` and ``trainIdx``
    indexes ``keypoints_destination``. RANSAC inlier thresholds and minimum
    acceptance criteria come from ``config.py``.

    Raises:
        TypeError: If inputs or config have an unexpected type.
        ValueError: If RANSAC parameters are invalid.
        HomographyEstimationError: If there are too few correspondences, RANSAC
            fails, or its output is degenerate or below configured thresholds.
    """

    if not isinstance(keypoints_source, (list, tuple)):
        raise TypeError("keypoints_source must be a list or tuple of OpenCV keypoints.")
    if not isinstance(keypoints_destination, (list, tuple)):
        raise TypeError("keypoints_destination must be a list or tuple of OpenCV keypoints.")
    if not isinstance(matches, (list, tuple)):
        raise TypeError("matches must be a list or tuple of OpenCV DMatch objects.")

    selected_config = DEFAULT_CONFIG.ransac if config is None else config
    if not isinstance(selected_config, RansacConfig):
        raise TypeError("config must be a RansacConfig instance.")
    if selected_config.reprojection_threshold <= 0:
        raise ValueError("reprojection_threshold must be greater than zero.")
    if selected_config.max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")
    if not 0.0 < selected_config.confidence < 1.0:
        raise ValueError("confidence must be greater than zero and less than one.")
    if selected_config.minimum_inliers < 4:
        raise ValueError("minimum_inliers must be at least four for a homography.")
    if not 0.0 <= selected_config.minimum_inlier_ratio <= 1.0:
        raise ValueError("minimum_inlier_ratio must be between zero and one.")

    total_matches = len(matches)
    if total_matches < 4:
        raise HomographyEstimationError(
            f"At least 4 point matches are required; received {total_matches}."
        )

    source_points = np.empty((total_matches, 2), dtype=np.float32)
    destination_points = np.empty((total_matches, 2), dtype=np.float32)
    for index, match in enumerate(matches):
        query_index = int(match.queryIdx)
        train_index = int(match.trainIdx)
        if query_index < 0 or query_index >= len(keypoints_source):
            raise HomographyEstimationError(
                f"Match {index} has queryIdx {query_index} outside the source keypoints."
            )
        if train_index < 0 or train_index >= len(keypoints_destination):
            raise HomographyEstimationError(
                f"Match {index} has trainIdx {train_index} outside the destination keypoints."
            )
        try:
            source_points[index] = keypoints_source[query_index].pt
            destination_points[index] = keypoints_destination[train_index].pt
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(f"Invalid keypoint or match at correspondence {index}.") from exc

    if not np.isfinite(source_points).all() or not np.isfinite(destination_points).all():
        raise HomographyEstimationError("Correspondence coordinates contain non-finite values.")

    H, cv_inlier_mask = cv2.findHomography(
        source_points,
        destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=selected_config.reprojection_threshold,
        maxIters=selected_config.max_iterations,
        confidence=selected_config.confidence,
    )
    if H is None:
        raise HomographyEstimationError("cv2.findHomography returned no homography matrix.")
    if not isinstance(H, np.ndarray) or H.shape != (3, 3):
        raise HomographyEstimationError(
            f"Expected a 3x3 homography matrix; got {getattr(H, 'shape', None)}."
        )
    H = np.asarray(H, dtype=np.float64)
    if not np.isfinite(H).all():
        raise HomographyEstimationError("Homography matrix contains non-finite values.")

    scale = H[2, 2]
    if abs(scale) > np.finfo(np.float64).eps:
        H = H / scale
    else:
        norm = float(np.linalg.norm(H))
        if norm <= np.finfo(np.float64).eps:
            raise HomographyEstimationError("Homography matrix is the zero matrix.")
        H = H / norm
    if np.linalg.matrix_rank(H) < 3 or abs(float(np.linalg.det(H))) <= 1e-12:
        raise HomographyEstimationError("Estimated homography matrix is singular (degenerate).")

    if cv_inlier_mask is None:
        raise HomographyEstimationError("cv2.findHomography did not return an inlier mask.")
    inlier_mask = np.asarray(cv_inlier_mask).reshape(-1).astype(bool)
    if inlier_mask.size != total_matches:
        raise HomographyEstimationError(
            f"Inlier mask length {inlier_mask.size} does not match {total_matches} matches."
        )

    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / total_matches
    if inlier_count < selected_config.minimum_inliers:
        raise HomographyEstimationError(
            f"Only {inlier_count} inliers; at least {selected_config.minimum_inliers} required."
        )
    if inlier_ratio < selected_config.minimum_inlier_ratio:
        raise HomographyEstimationError(
            f"Inlier ratio {inlier_ratio:.3f} is below configured minimum "
            f"{selected_config.minimum_inlier_ratio:.3f}."
        )

    source_inliers = source_points[inlier_mask].reshape(-1, 1, 2)
    destination_inliers = destination_points[inlier_mask]
    projected = cv2.perspectiveTransform(source_inliers, H).reshape(-1, 2)
    if not np.isfinite(projected).all():
        raise HomographyEstimationError("Homography produced non-finite projected points.")
    errors = np.linalg.norm(projected - destination_inliers, axis=1)
    reprojection_error = float(np.mean(errors))
    if not np.isfinite(reprojection_error):
        raise HomographyEstimationError("Reprojection error is not finite.")

    return HomographyResult(
        H=H,
        inlier_mask=inlier_mask,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        reprojection_error=reprojection_error,
        total_matches=total_matches,
    )


def visualize_homography_matches(
    image_source: NDArray[np.uint8],
    keypoints_source: list[Any],
    image_destination: NDArray[np.uint8],
    keypoints_destination: list[Any],
    matches: list[Any],
    result: HomographyResult,
    method: FeatureMethod,
    *,
    output_dir: str | Path | None = None,
    filename: str = "homography_01_02.jpg",
) -> Path:
    """Save side-by-side correspondences (green inliers, red outliers).

    Output is written under ``outputs/sift`` or ``outputs/orb`` by default and
    reopened once to confirm the saved image is decodable.
    """

    normalized_method = _normalize_method(method)
    for name, image in (("image_source", image_source), ("image_destination", image_destination)):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"{name} must be a non-empty 3-channel BGR image.")
        if image.dtype != np.uint8:
            raise TypeError(f"{name} must have dtype uint8.")
    if not isinstance(result, HomographyResult):
        raise TypeError("result must be a HomographyResult instance.")
    if len(matches) != result.total_matches or result.inlier_mask.size != len(matches):
        raise ValueError("Matches and homography inlier mask must have the same length.")
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a non-empty file name without directory parts.")

    top_margin = 34
    canvas_height = top_margin + max(image_source.shape[0], image_destination.shape[0])
    canvas_width = image_source.shape[1] + image_destination.shape[1]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[top_margin : top_margin + image_source.shape[0], : image_source.shape[1]] = image_source
    destination_x = image_source.shape[1]
    canvas[
        top_margin : top_margin + image_destination.shape[0],
        destination_x : destination_x + image_destination.shape[1],
    ] = image_destination

    cv2.putText(canvas, "Inliers: green", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    cv2.putText(canvas, "Outliers: red", (190, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for match, is_inlier in zip(matches, result.inlier_mask, strict=True):
        try:
            source_point = keypoints_source[int(match.queryIdx)].pt
            destination_point = keypoints_destination[int(match.trainIdx)].pt
        except (IndexError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Match indices do not correspond to the supplied keypoints.") from exc
        color = (0, 220, 0) if is_inlier else (0, 0, 255)
        source_xy = (int(round(source_point[0])), int(round(source_point[1])) + top_margin)
        destination_xy = (
            int(round(destination_point[0])) + destination_x,
            int(round(destination_point[1])) + top_margin,
        )
        cv2.line(canvas, source_xy, destination_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, source_xy, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, destination_xy, 3, color, -1, cv2.LINE_AA)

    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_CONFIG.output_dir / normalized_method
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / filename
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not save homography visualization: {output_path}")
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if reopened is None or reopened.size == 0:
        raise OSError(f"Saved homography visualization could not be reopened: {output_path}")
    return output_path

