"""End-to-end tests for each feature method on the four-image sample set."""

from __future__ import annotations

import csv

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

    with (result.output_dir / "evaluation.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert len([row for row in records if row["record_type"] == "image"]) == 4
    assert len([row for row in records if row["record_type"] == "pair"]) == 3
    assert len([row for row in records if row["record_type"] == "summary"]) == 1

