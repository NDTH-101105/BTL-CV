"""Mask-aware feather blending for images on a shared canvas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BlendResult:
    """Blended BGR panorama and masks describing its pixel coverage."""

    panorama: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    overlap_mask: NDArray[np.uint8]
    coverage_count: NDArray[np.uint16]


def blend_images(
    images: Sequence[NDArray[np.uint8]],
    masks: Sequence[NDArray[np.uint8]],
) -> BlendResult:
    """Blend valid pixels, feathering only where two or more masks overlap.

    Images and masks may differ in size; they are top-left aligned and padded
    to the largest height and width. Pixels supported by only one mask are
    copied exactly. Empty masks contribute no pixels, and black image values
    remain valid wherever their mask says they are valid.
    """

    if len(images) != len(masks):
        raise ValueError(f"images and masks must have equal lengths; got {len(images)} and {len(masks)}.")
    if not images:
        raise ValueError("At least one image and mask pair is required.")

    checked_images: list[NDArray[np.uint8]] = []
    checked_masks: list[NDArray[np.bool_]] = []
    max_height = 0
    max_width = 0
    for index, (image, mask) in enumerate(zip(images, masks, strict=True)):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"images[{index}] must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"images[{index}] must be a uint8 BGR image with shape (H, W, 3).")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"masks[{index}] must be a NumPy array.")
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.ndim != 2:
            raise ValueError(f"masks[{index}] must be a 2-D mask; got shape {mask.shape}.")
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"masks[{index}] shape {mask.shape} does not match image shape {image.shape[:2]}."
            )
        checked_images.append(image)
        checked_masks.append(mask > 0)
        max_height = max(max_height, image.shape[0])
        max_width = max(max_width, image.shape[1])

    if max_height == 0 or max_width == 0:
        raise ValueError("Input images must have non-zero width and height.")

    coverage = np.zeros((max_height, max_width), dtype=np.uint16)
    for image, valid in zip(checked_images, checked_masks, strict=True):
        height, width = image.shape[:2]
        coverage[:height, :width] += valid.astype(np.uint16)

    output = np.zeros((max_height, max_width, 3), dtype=np.uint8)
    # Assign exclusive areas directly so their pixel values remain unchanged.
    for image, valid in zip(checked_images, checked_masks, strict=True):
        height, width = image.shape[:2]
        exclusive = valid & (coverage[:height, :width] == 1)
        target = output[:height, :width]
        target[exclusive] = image[exclusive]

    overlap = coverage >= 2
    if np.any(overlap):
        # Limit the floating-point accumulators to the overlap bounding box.
        # The panorama canvas can be much larger than its valid content.
        overlap_rows = np.flatnonzero(np.any(overlap, axis=1))
        overlap_columns = np.flatnonzero(np.any(overlap, axis=0))
        overlap_y0, overlap_y1 = int(overlap_rows[0]), int(overlap_rows[-1]) + 1
        overlap_x0, overlap_x1 = int(overlap_columns[0]), int(overlap_columns[-1]) + 1
        overlap_roi = overlap[overlap_y0:overlap_y1, overlap_x0:overlap_x1]
        roi_height, roi_width = overlap_roi.shape
        accumulated = np.zeros((roi_height, roi_width, 3), dtype=np.float32)
        total_weight = np.zeros((roi_height, roi_width), dtype=np.float32)

        for image, valid in zip(checked_images, checked_masks, strict=True):
            height, width = image.shape[:2]
            mask_rows = np.flatnonzero(np.any(valid, axis=1))
            mask_columns = np.flatnonzero(np.any(valid, axis=0))
            if mask_rows.size == 0 or mask_columns.size == 0:
                continue
            mask_y0, mask_y1 = int(mask_rows[0]), int(mask_rows[-1]) + 1
            mask_x0, mask_x1 = int(mask_columns[0]), int(mask_columns[-1]) + 1

            # Padding with one invalid pixel keeps distance-to-mask-edge values
            # equivalent to running the distance transform on the full canvas.
            local_valid = valid[mask_y0:mask_y1, mask_x0:mask_x1]
            padded_valid = cv2.copyMakeBorder(
                local_valid.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
            )
            distance = cv2.distanceTransform(padded_valid, cv2.DIST_L2, 3)[1:-1, 1:-1]

            y0, y1 = max(overlap_y0, mask_y0), min(overlap_y1, mask_y1, height)
            x0, x1 = max(overlap_x0, mask_x0), min(overlap_x1, mask_x1, width)
            if y0 >= y1 or x0 >= x1:
                continue

            source_valid = valid[y0:y1, x0:x1]
            local_overlap = overlap[y0:y1, x0:x1]
            weights = np.maximum(
                distance[y0 - mask_y0 : y1 - mask_y0, x0 - mask_x0 : x1 - mask_x0],
                1.0,
            )
            weights *= source_valid & local_overlap

            target_y = slice(y0 - overlap_y0, y1 - overlap_y0)
            target_x = slice(x0 - overlap_x0, x1 - overlap_x0)
            source = image[y0:y1, x0:x1]
            # Channel-wise float32 operations avoid a full-canvas float64 RGB
            # temporary while preserving the same feather weighting rule.
            for channel in range(3):
                accumulated[target_y, target_x, channel] += (
                    source[:, :, channel].astype(np.float32) * weights
                )
            total_weight[target_y, target_x] += weights

        overlap_weights = total_weight[overlap_roi]
        if np.any(overlap_weights <= 0) or not np.isfinite(overlap_weights).all():
            raise RuntimeError("Feather blending produced invalid weights in an overlap region.")
        output_roi = output[overlap_y0:overlap_y1, overlap_x0:overlap_x1]
        blended = np.rint(accumulated[overlap_roi] / overlap_weights[:, None])
        output_roi[overlap_roi] = np.clip(blended, 0, 255).astype(np.uint8)

    valid_mask = (coverage > 0).astype(np.uint8) * 255
    overlap_mask = overlap.astype(np.uint8) * 255
    return BlendResult(
        panorama=output,
        mask=valid_mask,
        overlap_mask=overlap_mask,
        coverage_count=coverage,
    )


def save_blend_result(
    result: BlendResult,
    output_dir: str | Path,
    filename: str = "panorama_blended.png",
) -> list[Path]:
    """Save the blended panorama and its coverage/overlap masks."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rgba = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = result.mask
    artifacts = (
        (filename, result.panorama),
        ("panorama_transparent.png", rgba),
        ("panorama_valid_mask.png", result.mask),
        ("panorama_overlap_mask.png", result.overlap_mask),
    )
    saved: list[Path] = []
    for name, image in artifacts:
        path = directory / name
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not save blending result: {path}")
        reopened = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if reopened is None or reopened.size == 0:
            raise OSError(f"Saved blending result could not be reopened: {path}")
        saved.append(path)
    return saved

