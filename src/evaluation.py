"""Experiment metrics and CSV reporting for the stitching pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ImageMetrics:
    """Feature count and extraction time for one input image."""

    index: int
    image_name: str
    keypoints: int
    feature_seconds: float


@dataclass(frozen=True)
class PairMetrics:
    """Matching and homography statistics for an adjacent image pair."""

    source_image: str
    destination_image: str
    knn_candidates: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    reprojection_error: float
    matching_seconds: float
    homography_seconds: float


@dataclass(frozen=True)
class PipelineTimings:
    """Wall-clock timings for the main pipeline stages, in seconds."""

    image_loading: float
    feature_extraction: float
    matching: float
    homography: float
    warping: float
    blending: float
    total: float


@dataclass(frozen=True)
class PipelineResult:
    """Serializable experiment summary returned after a successful panorama."""

    method: str
    reference_image: str
    output_dir: Path
    image_metrics: tuple[ImageMetrics, ...]
    pair_metrics: tuple[PairMetrics, ...]
    timings: PipelineTimings
    canvas_width: int
    canvas_height: int
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class PipelineFailure:
    """A method/pair failure recorded by the command-line runner."""

    method: str
    stage: str
    reason: str
    source_image: str = ""
    destination_image: str = ""


CSV_FIELDS = (
    "method",
    "record_type",
    "image_index",
    "image_name",
    "source_image",
    "destination_image",
    "reference_image",
    "keypoints",
    "source_keypoints",
    "destination_keypoints",
    "feature_seconds",
    "source_feature_seconds",
    "destination_feature_seconds",
    "knn_candidates",
    "good_matches",
    "inliers",
    "inlier_ratio",
    "reprojection_error_px",
    "matching_seconds",
    "homography_seconds",
    "image_loading_seconds",
    "feature_extraction_seconds_total",
    "matching_seconds_total",
    "homography_seconds_total",
    "warping_seconds",
    "blending_seconds",
    "pipeline_seconds_total",
    "canvas_width",
    "canvas_height",
    "success",
    "stage",
    "reason",
    "output_dir",
)


def write_evaluation_csv(
    results: Sequence[PipelineResult],
    output_path: str | Path,
    failures: Sequence[PipelineFailure] = (),
) -> Path:
    """Write image-, pair-, summary-, and failure-level records to a CSV file.

    The CSV contains measurements only. Human visual observations should be
    recorded separately in the report and are not inferred from match counts.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for result in results:
        common = {
            "method": result.method,
            "success": True,
            "reference_image": result.reference_image,
            "output_dir": str(result.output_dir),
            "canvas_width": result.canvas_width,
            "canvas_height": result.canvas_height,
        }
        images_by_name = {metric.image_name: metric for metric in result.image_metrics}
        for metric in result.image_metrics:
            rows.append(
                {
                    **common,
                    "record_type": "image",
                    "image_index": metric.index,
                    "image_name": metric.image_name,
                    "keypoints": metric.keypoints,
                    "feature_seconds": metric.feature_seconds,
                }
            )
        for pair in result.pair_metrics:
            source_metric = images_by_name[pair.source_image]
            destination_metric = images_by_name[pair.destination_image]
            rows.append(
                {
                    **common,
                    "record_type": "pair",
                    "source_image": pair.source_image,
                    "destination_image": pair.destination_image,
                    "source_keypoints": source_metric.keypoints,
                    "destination_keypoints": destination_metric.keypoints,
                    "source_feature_seconds": source_metric.feature_seconds,
                    "destination_feature_seconds": destination_metric.feature_seconds,
                    "knn_candidates": pair.knn_candidates,
                    "good_matches": pair.good_matches,
                    "inliers": pair.inliers,
                    "inlier_ratio": pair.inlier_ratio,
                    "reprojection_error_px": pair.reprojection_error,
                    "matching_seconds": pair.matching_seconds,
                    "homography_seconds": pair.homography_seconds,
                }
            )
        timing = result.timings
        rows.append(
            {
                **common,
                "record_type": "summary",
                "image_loading_seconds": timing.image_loading,
                "feature_extraction_seconds_total": timing.feature_extraction,
                "matching_seconds_total": timing.matching,
                "homography_seconds_total": timing.homography,
                "warping_seconds": timing.warping,
                "blending_seconds": timing.blending,
                "pipeline_seconds_total": timing.total,
            }
        )

    for failure in failures:
        rows.append(
            {
                "method": failure.method,
                "record_type": "failure",
                "source_image": failure.source_image,
                "destination_image": failure.destination_image,
                "success": False,
                "stage": failure.stage,
                "reason": failure.reason,
            }
        )

    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path

