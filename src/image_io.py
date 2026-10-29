"""Image loading and preprocessing helpers for image stitching.

Images are returned in filename order. Their original dimensions are kept;
no resizing or other geometric transform is applied. Color arrays use OpenCV's
BGR channel order, and grayscale arrays are derived from those same pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from src.config import DEFAULT_CONFIG


SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class ImageLoadError(RuntimeError):
    """Raised when an input directory or one of its images cannot be loaded."""


@dataclass(frozen=True)
class LoadedImage:
    """An input image and its grayscale view, with the source path retained."""

    path: Path
    color: np.ndarray
    gray: np.ndarray


def _get_cv2():
    """Import OpenCV lazily so callers receive an actionable dependency error."""

    try:
        import cv2
    except ImportError as exc:
        raise ImageLoadError(
            "OpenCV is required to load images. Install it with "
            "`python -m pip install opencv-python`."
        ) from exc
    return cv2


def load_images(
    directory: str | Path = DEFAULT_CONFIG.data_dir,
    *,
    extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
) -> list[LoadedImage]:
    """Load supported images in case-insensitive filename order.

    Args:
        directory: Folder containing one panorama sequence.
        extensions: Supported filename suffixes, case-insensitive.

    Returns:
        Loaded images in deterministic filename order. Color images remain BGR
        and grayscale versions are generated for feature detection.

    Raises:
        ImageLoadError: If the folder is missing/not a directory, contains no
            supported images, OpenCV is unavailable, or a file cannot be read.
    """

    image_dir = Path(directory).expanduser()
    if not image_dir.exists():
        raise ImageLoadError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise ImageLoadError(f"Image path is not a directory: {image_dir}")

    normalized_extensions = frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )
    paths = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in normalized_extensions
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not paths:
        suffixes = ", ".join(sorted(normalized_extensions))
        raise ImageLoadError(
            f"No supported images found in {image_dir}. Supported extensions: {suffixes}"
        )

    cv2 = _get_cv2()
    loaded: list[LoadedImage] = []
    for path in paths:
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None or color.size == 0:
            raise ImageLoadError(f"Could not read image file: {path}")
        if color.ndim != 3 or color.shape[2] != 3:
            raise ImageLoadError(
                f"Expected a 3-channel color image at {path}; got shape {color.shape}"
            )

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if gray is None or gray.size == 0:
            raise ImageLoadError(f"Could not convert image to grayscale: {path}")
        loaded.append(LoadedImage(path=path, color=color, gray=gray))

    return loaded

