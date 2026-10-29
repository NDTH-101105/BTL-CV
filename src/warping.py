"""Project images into a common canvas using source-to-destination homographies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray


class WarpingError(RuntimeError):
    """Raised when image transforms cannot be safely projected to a canvas."""


@dataclass(frozen=True)
class WarpedImage:
    """One color image and its valid-pixel mask on the shared panorama canvas."""

    image: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    source_to_reference: NDArray[np.float64]
    source_to_canvas: NDArray[np.float64]


@dataclass(frozen=True)
class WarpedPanorama:
    """All images warped to one common canvas, before blending."""

    images: list[WarpedImage]
    reference_index: int
    canvas_size: tuple[int, int]  # (width, height)
    canvas_translation: NDArray[np.float64]
    bounds: tuple[int, int, int, int]  # (min_x, min_y, max_x, max_y), inclusive


def _validated_homography(matrix: NDArray[np.generic], label: str) -> NDArray[np.float64]:
    """Return a normalized, finite, invertible 3x3 matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"{label} must have shape (3, 3); got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values.")
    scale = value[2, 2]
    if abs(scale) > np.finfo(np.float64).eps:
        value = value / scale
    else:
        norm = float(np.linalg.norm(value))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError(f"{label} is the zero matrix.")
        value = value / norm
    if np.linalg.matrix_rank(value) < 3 or abs(float(np.linalg.det(value))) <= 1e-12:
        raise ValueError(f"{label} is singular and cannot be used for warping.")
    return value


def transforms_to_reference(
    image_count: int,
    pairwise_homographies: Sequence[NDArray[np.generic]],
    reference_index: int = 0,
) -> list[NDArray[np.float64]]:
    """Compose adjacent transforms into source-image-to-reference transforms.

    Each ``pairwise_homographies[i]`` maps coordinates in image ``i`` to
    coordinates in image ``i + 1``. If ``H_i`` maps i→i+1 and ``T_i`` maps i
    to the reference, composition uses ``T_i = T_(i+1) @ H_i`` to the left of
    the reference and ``T_(i+1) = T_i @ inverse(H_i)`` to its right.
    """

    if image_count < 1:
        raise ValueError("image_count must be at least one.")
    if not 0 <= reference_index < image_count:
        raise ValueError(f"reference_index must be in [0, {image_count - 1}].")
    if len(pairwise_homographies) != image_count - 1:
        raise ValueError(
            f"Expected {image_count - 1} adjacent homographies for {image_count} images; "
            f"received {len(pairwise_homographies)}."
        )

    adjacent = [
        _validated_homography(matrix, f"pairwise_homographies[{index}]")
        for index, matrix in enumerate(pairwise_homographies)
    ]
    transforms: list[NDArray[np.float64] | None] = [None] * image_count
    transforms[reference_index] = np.eye(3, dtype=np.float64)

    # H_i maps i to i+1, so left-compose it onto the transform already mapping
    # image i+1 into the reference coordinate system.
    for index in range(reference_index - 1, -1, -1):
        transforms[index] = transforms[index + 1] @ adjacent[index]  # type: ignore[operator]

    # To travel from image i+1 back to i, invert H_i, then apply T_i.
    for index in range(reference_index, image_count - 1):
        try:
            inverse = np.linalg.inv(adjacent[index])
        except np.linalg.LinAlgError as exc:
            raise WarpingError(f"Adjacent homography {index} could not be inverted.") from exc
        transforms[index + 1] = transforms[index] @ inverse  # type: ignore[operator]

    return [
        _validated_homography(transform, f"transform for image {index}")
        for index, transform in enumerate(transforms)
        if transform is not None
    ]


def _project_corners(
    image: NDArray[np.uint8], transform: NDArray[np.float64], index: int
) -> NDArray[np.float64]:
    """Project pixel-center corners, rejecting a projective horizon crossing."""

    height, width = image.shape[:2]
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]],
        dtype=np.float64,
    ).T
    projected_h = transform @ corners
    denominators = projected_h[2]
    if np.any(np.abs(denominators) <= 1e-10) or (
        np.min(denominators) < 0 < np.max(denominators)
    ):
        raise WarpingError(f"Image {index} transform crosses the projective horizon.")
    projected = (projected_h[:2] / denominators).T
    if not np.isfinite(projected).all():
        raise WarpingError(f"Image {index} has non-finite projected corners.")
    return projected


def warp_images_to_reference(
    images: Sequence[NDArray[np.uint8]],
    pairwise_homographies: Sequence[NDArray[np.generic]],
    reference_index: int = 0,
    *,
    max_canvas_pixels: int = 100_000_000,
) -> WarpedPanorama:
    """Warp BGR images and validity masks onto a shared reference canvas.

    The pairwise matrices must map image i to image i+1, matching
    ``HomographyResult.H``. The reference image is left unchanged apart from
    the common canvas translation needed to handle negative coordinates.
    """

    if not isinstance(images, (list, tuple)) or not images:
        raise ValueError("images must be a non-empty list or tuple.")
    if max_canvas_pixels <= 0:
        raise ValueError("max_canvas_pixels must be greater than zero.")
    checked_images: list[NDArray[np.uint8]] = []
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"images[{index}] must be a NumPy array.")
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"images[{index}] must be a non-empty 3-channel BGR image.")
        if image.dtype != np.uint8:
            raise TypeError(f"images[{index}] must have dtype uint8.")
        checked_images.append(image)

    reference_transforms = transforms_to_reference(
        len(checked_images), pairwise_homographies, reference_index
    )
    all_corners = [
        _project_corners(image, transform, index)
        for index, (image, transform) in enumerate(zip(checked_images, reference_transforms, strict=True))
    ]
    corners = np.concatenate(all_corners, axis=0)
    min_x, min_y = np.floor(corners.min(axis=0)).astype(np.int64)
    max_x, max_y = np.ceil(corners.max(axis=0)).astype(np.int64)
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)
    if width <= 0 or height <= 0:
        raise WarpingError(f"Computed invalid canvas size {(width, height)}.")
    if width * height > max_canvas_pixels:
        raise WarpingError(
            f"Canvas {width}x{height} ({width * height:,} pixels) exceeds limit "
            f"{max_canvas_pixels:,}. Check homographies or raise max_canvas_pixels."
        )

    translation = np.array(
        [[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    canvas_images: list[WarpedImage] = []
    for image, reference_transform in zip(checked_images, reference_transforms, strict=True):
        source_to_canvas = translation @ reference_transform
        warped_image = cv2.warpPerspective(
            image,
            source_to_canvas,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        warped_mask = cv2.warpPerspective(
            source_mask,
            source_to_canvas,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        canvas_images.append(
            WarpedImage(
                image=warped_image,
                mask=warped_mask,
                source_to_reference=reference_transform,
                source_to_canvas=source_to_canvas,
            )
        )

    return WarpedPanorama(
        images=canvas_images,
        reference_index=reference_index,
        canvas_size=(width, height),
        canvas_translation=translation,
        bounds=(int(min_x), int(min_y), int(max_x), int(max_y)),
    )


def save_warped_images(warped: WarpedPanorama, output_dir: str | Path) -> list[Path]:
    """Save each warped BGR image and its valid-pixel mask as PNG files."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, item in enumerate(warped.images):
        for suffix, array in (("image", item.image), ("mask", item.mask)):
            path = directory / f"warped_{index:02d}_{suffix}.png"
            if not cv2.imwrite(str(path), array):
                raise OSError(f"Could not save warped {suffix}: {path}")
            reopened = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if reopened is None or reopened.size == 0:
                raise OSError(f"Saved warped {suffix} could not be reopened: {path}")
            saved.append(path)
    return saved

