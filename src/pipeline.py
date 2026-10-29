"""End-to-end orchestration for one image feature method."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from src.blending import blend_images, save_blend_result
from src.config import DEFAULT_CONFIG
from src.evaluation import ImageMetrics, PairMetrics, PipelineResult, PipelineTimings, write_evaluation_csv
from src.features import FeatureMethod, _normalize_method, extract_features
from src.homography import HomographyEstimationError, estimate_homography, visualize_homography_matches
from src.image_io import LoadedImage, load_images
from src.matching import MatchResult, match_descriptors, visualize_matches
from src.warping import save_warped_images, warp_images_to_reference


class PipelineError(RuntimeError):
    """A stage failure with method and adjacent-pair context."""

    def __init__(
        self,
        method: str,
        stage: str,
        reason: str,
        source_image: str = "",
        destination_image: str = "",
    ) -> None:
        self.method = method
        self.stage = stage
        self.reason = reason
        self.source_image = source_image
        self.destination_image = destination_image
        pair = f" pair {source_image}–{destination_image}" if source_image else ""
        super().__init__(f"{method.upper()}{pair} failed during {stage}: {reason}")


def _save_unblended_preview(images: list[Any], output_path: Path) -> Path:
    """Save a first-valid-pixel composite to show the seam before blending."""

    if not images:
        raise ValueError("At least one warped image is required for a preview.")
    height, width = images[0].image.shape[:2]
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    occupied = np.zeros((height, width), dtype=bool)
    for item in images:
        valid = item.mask > 0
        take = valid & ~occupied
        preview[take] = item.image[take]
        occupied |= valid
    if not cv2.imwrite(str(output_path), preview):
        raise OSError(f"Could not save pre-blending preview: {output_path}")
    reopened = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    if reopened is None or reopened.size == 0:
        raise OSError(f"Saved pre-blending preview could not be reopened: {output_path}")
    return output_path


def _pair_label(source: LoadedImage, destination: LoadedImage) -> str:
    return f"{source.path.stem}_{destination.path.stem}"


def run_pipeline(
    method: FeatureMethod,
    image_dir: str | Path = DEFAULT_CONFIG.data_dir,
    output_root: str | Path = DEFAULT_CONFIG.output_dir,
    reference_index: int | None = None,
) -> PipelineResult:
    """Run the existing load→features→matches→homographies→warp→blend stages.

    Each input image's descriptors are computed once. Only adjacent image pairs
    are matched and estimated. Pairwise homographies retain the established
    source-to-destination convention and are composed by ``warping.py``.
    Any match or homography failure aborts this method before panorama creation.
    """

    normalized_method = _normalize_method(method)
    pipeline_start = perf_counter()
    load_start = perf_counter()
    images = load_images(image_dir)
    loading_seconds = perf_counter() - load_start
    if len(images) < 2:
        raise PipelineError(normalized_method, "image loading", "At least two images are required.")
    if reference_index is None:
        reference_index = (len(images) - 1) // 2
    if not 0 <= reference_index < len(images):
        raise ValueError(f"reference_index must be in [0, {len(images) - 1}].")

    method_output_dir = Path(output_root) / normalized_method
    method_output_dir.mkdir(parents=True, exist_ok=True)

    keypoints_by_image: list[list[Any]] = []
    descriptors_by_image: list[np.ndarray | None] = []
    image_metrics: list[ImageMetrics] = []
    feature_total = 0.0
    for index, image in enumerate(images):
        start = perf_counter()
        try:
            keypoints, descriptors = extract_features(image.gray, normalized_method)  # type: ignore[arg-type]
        except (ValueError, TypeError, RuntimeError) as exc:
            raise PipelineError(normalized_method, "feature extraction", f"{image.path.name}: {exc}") from exc
        elapsed = perf_counter() - start
        feature_total += elapsed
        if descriptors is None or not keypoints:
            raise PipelineError(
                normalized_method,
                "feature extraction",
                f"No descriptors found in {image.path.name}.",
            )
        keypoints_by_image.append(keypoints)
        descriptors_by_image.append(descriptors)
        image_metrics.append(ImageMetrics(index, image.path.name, len(keypoints), elapsed))

    homographies: list[np.ndarray] = []
    pair_metrics: list[PairMetrics] = []
    pair_match_results: list[MatchResult] = []
    matching_total = 0.0
    homography_total = 0.0
    for index in range(len(images) - 1):
        source = images[index]
        destination = images[index + 1]
        label = _pair_label(source, destination)
        match_start = perf_counter()
        try:
            match_result = match_descriptors(
                descriptors_by_image[index],
                descriptors_by_image[index + 1],
                normalized_method,  # type: ignore[arg-type]
            )
        except (ValueError, TypeError, cv2.error) as exc:
            raise PipelineError(
                normalized_method, "matching", str(exc), source.path.name, destination.path.name
            ) from exc
        matching_seconds = perf_counter() - match_start
        matching_total += matching_seconds
        if not match_result.statistics.has_enough_matches:
            raise PipelineError(
                normalized_method,
                "matching",
                f"{match_result.statistics.good_matches} good matches; "
                f"{match_result.statistics.minimum_required} required "
                f"(status={match_result.statistics.status}).",
                source.path.name,
                destination.path.name,
            )

        homography_start = perf_counter()
        try:
            homography = estimate_homography(
                keypoints_by_image[index], keypoints_by_image[index + 1], match_result.matches
            )
        except (HomographyEstimationError, ValueError, TypeError, cv2.error) as exc:
            raise PipelineError(
                normalized_method, "homography", str(exc), source.path.name, destination.path.name
            ) from exc
        homography_seconds = perf_counter() - homography_start
        homography_total += homography_seconds

        try:
            visualize_matches(
                source.color,
                keypoints_by_image[index],
                destination.color,
                keypoints_by_image[index + 1],
                match_result,
                normalized_method,  # type: ignore[arg-type]
                output_dir=method_output_dir,
                filename=f"matches_{label}.jpg",
            )
            visualize_homography_matches(
                source.color,
                keypoints_by_image[index],
                destination.color,
                keypoints_by_image[index + 1],
                match_result.matches,
                homography,
                normalized_method,  # type: ignore[arg-type]
                output_dir=method_output_dir,
                filename=f"homography_{label}.jpg",
            )
        except (ValueError, TypeError, OSError, cv2.error) as exc:
            raise PipelineError(
                normalized_method, "visualization", str(exc), source.path.name, destination.path.name
            ) from exc

        homographies.append(homography.H)
        pair_match_results.append(match_result)
        pair_metrics.append(
            PairMetrics(
                source_image=source.path.name,
                destination_image=destination.path.name,
                knn_candidates=match_result.statistics.knn_candidates,
                good_matches=match_result.statistics.good_matches,
                inliers=homography.inlier_count,
                inlier_ratio=homography.inlier_ratio,
                reprojection_error=homography.reprojection_error,
                matching_seconds=matching_seconds,
                homography_seconds=homography_seconds,
            )
        )

    warp_start = perf_counter()
    try:
        warped = warp_images_to_reference(
            [image.color for image in images], homographies, reference_index=reference_index
        )
        save_warped_images(warped, method_output_dir)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise PipelineError(normalized_method, "warping", str(exc)) from exc
    warping_seconds = perf_counter() - warp_start

    blending_start = perf_counter()
    try:
        _save_unblended_preview(
            warped.images, method_output_dir / "panorama_before_blending.png"
        )
        blended = blend_images(
            [item.image for item in warped.images], [item.mask for item in warped.images]
        )
        save_blend_result(blended, method_output_dir)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        raise PipelineError(normalized_method, "blending", str(exc)) from exc
    blending_seconds = perf_counter() - blending_start
    total_seconds = perf_counter() - pipeline_start

    artifacts = {
        "panorama_before_blending": method_output_dir / "panorama_before_blending.png",
        "panorama_blended": method_output_dir / "panorama_blended.png",
        "valid_mask": method_output_dir / "panorama_valid_mask.png",
        "overlap_mask": method_output_dir / "panorama_overlap_mask.png",
    }
    for index in range(len(images)):
        artifacts[f"warped_image_{index:02d}"] = method_output_dir / f"warped_{index:02d}_image.png"
        artifacts[f"warped_mask_{index:02d}"] = method_output_dir / f"warped_{index:02d}_mask.png"
    for index in range(len(pair_metrics)):
        label = f"{images[index].path.stem}_{images[index + 1].path.stem}"
        artifacts[f"matches_{label}"] = method_output_dir / f"matches_{label}.jpg"
        artifacts[f"homography_{label}"] = method_output_dir / f"homography_{label}.jpg"

    result = PipelineResult(
        method=normalized_method,
        reference_image=images[reference_index].path.name,
        output_dir=method_output_dir,
        image_metrics=tuple(image_metrics),
        pair_metrics=tuple(pair_metrics),
        timings=PipelineTimings(
            image_loading=loading_seconds,
            feature_extraction=feature_total,
            matching=matching_total,
            homography=homography_total,
            warping=warping_seconds,
            blending=blending_seconds,
            total=total_seconds,
        ),
        canvas_width=warped.canvas_size[0],
        canvas_height=warped.canvas_size[1],
        artifacts=artifacts,
    )
    write_evaluation_csv([result], method_output_dir / "evaluation.csv")
    return result

