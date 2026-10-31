"""End-to-end tests for each feature method on the four-image sample set."""

from __future__ import annotations

import csv
import math

import cv2
import pytest

from src.config import DEFAULT_CONFIG
from src.pipeline import run_pipeline


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_complete_pipeline_on_boat_images(method: str, tmp_path) -> None:
    result = run_pipeline(method, DEFAULT_CONFIG.data_dir, tmp_path)

    assert result.method == method
    assert len(result.image_metrics) == 4
    assert len(result.pair_metrics) == 3
    assert [metric.image_name for metric in result.image_metrics] == [
        "01.jpg", "02.jpg", "03.jpg", "04.jpg"
    ]
    assert [(pair.source_image, pair.destination_image) for pair in result.pair_metrics] == [
        ("01.jpg", "02.jpg"), ("02.jpg", "03.jpg"), ("03.jpg", "04.jpg")
    ]
    assert result.reference_image == "02.jpg"
    panorama_path = result.artifacts["panorama_blended"]
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    assert panorama is not None and panorama.size > 0
    assert (panorama.shape[1], panorama.shape[0]) == (result.canvas_width, result.canvas_height)
    assert panorama.shape[0] >= 2592 and panorama.shape[1] >= 3888
    assert cv2.countNonZero(cv2.imread(str(result.artifacts["valid_mask"]), cv2.IMREAD_GRAYSCALE)) > 0
    assert all(path.is_file() for path in result.artifacts.values())
    transparent = cv2.imread(str(result.artifacts["panorama_transparent"]), cv2.IMREAD_UNCHANGED)
    assert transparent is not None and transparent.shape == (*panorama.shape[:2], 4)

    with (result.output_dir / "evaluation.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    image_rows = [row for row in records if row["record_type"] == "image"]
    pair_rows = [row for row in records if row["record_type"] == "pair"]
    summary_rows = [row for row in records if row["record_type"] == "summary"]
    assert len(image_rows) == 4
    assert len(pair_rows) == 3
    assert len(summary_rows) == 1
    assert not [row for row in records if row["record_type"] == "failure"]

    for row in image_rows:
        assert row["image_name"] and int(row["keypoints"]) > 0
        assert math.isfinite(float(row["feature_seconds"]))
    for row in pair_rows:
        good_matches = int(row["good_matches"])
        inliers = int(row["inliers"])
        assert good_matches >= inliers > 0
        assert float(row["inlier_ratio"]) == pytest.approx(inliers / good_matches)
        assert math.isfinite(float(row["reprojection_error_px"]))
        assert float(row["reprojection_error_px"]) >= 0
        for field in ("matching_seconds", "homography_seconds"):
            assert math.isfinite(float(row[field]))
    summary = summary_rows[0]
    timing_fields = (
        "image_loading_seconds",
        "feature_extraction_seconds_total",
        "matching_seconds_total",
        "homography_seconds_total",
        "warping_seconds",
        "blending_seconds",
        "pipeline_seconds_total",
    )
    for field in timing_fields:
        assert math.isfinite(float(summary[field]))
        assert float(summary[field]) >= 0
    assert float(summary["pipeline_seconds_total"]) >= sum(float(summary[k]) for k in timing_fields[:-1])

