"""Feature detection and description for SIFT and ORB."""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import DEFAULT_CONFIG, ORBConfig, SIFTConfig

FeatureMethod = Literal["sift", "orb"]
GrayImage = NDArray[np.uint8]


def _normalize_method(method: str) -> FeatureMethod:
    normalized = method.strip().lower()
    if normalized not in {"sift", "orb"}:
        raise ValueError(f"Unsupported feature method {method!r}; use 'sift' or 'orb'.")
    return normalized  # type: ignore[return-value]


def extract_features(
    gray_image: GrayImage,
    method: FeatureMethod,
    config: SIFTConfig | ORBConfig | None = None,
) -> tuple[list[Any], NDArray[Any] | None]:
    """Detect keypoints and compute descriptors for one grayscale image.

    Args:
        gray_image: Non-empty, single-channel 8-bit image. Its dimensions and
            coordinate system are preserved.
        method: ``"sift"`` or ``"orb"`` (case-insensitive at runtime).
        config: Method-specific parameters. Defaults to ``DEFAULT_CONFIG``.

    Returns:
        A list of OpenCV keypoints and a descriptor matrix. If no features are
        found, returns an empty keypoint list and ``None`` descriptors.

    Raises:
        TypeError: If the image or method-specific config has the wrong type.
        ValueError: If the image has invalid dimensions or is empty.
        RuntimeError: If the selected detector is unavailable in this OpenCV build.
    """

    if not isinstance(gray_image, np.ndarray):
        raise TypeError("gray_image must be a NumPy array.")
    if gray_image.ndim != 2:
        raise ValueError(f"Expected a 2-D grayscale image; got shape {gray_image.shape}.")
    if gray_image.size == 0 or gray_image.shape[0] == 0 or gray_image.shape[1] == 0:
        raise ValueError("gray_image must not be empty.")
    if gray_image.dtype != np.uint8:
        raise TypeError(f"gray_image must have dtype uint8; got {gray_image.dtype}.")

    normalized = _normalize_method(method)
    if normalized == "sift":
        selected_config = DEFAULT_CONFIG.sift if config is None else config
        if not isinstance(selected_config, SIFTConfig):
            raise TypeError("SIFT extraction requires an SIFTConfig instance.")
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("This OpenCV build does not provide cv2.SIFT_create.")
        detector = cv2.SIFT_create(
            nfeatures=selected_config.nfeatures,
            nOctaveLayers=selected_config.n_octave_layers,
            contrastThreshold=selected_config.contrast_threshold,
            edgeThreshold=selected_config.edge_threshold,
            sigma=selected_config.sigma,
        )
    else:
        selected_config = DEFAULT_CONFIG.orb if config is None else config
        if not isinstance(selected_config, ORBConfig):
            raise TypeError("ORB extraction requires an ORBConfig instance.")
        detector = cv2.ORB_create(
            nfeatures=selected_config.nfeatures,
            scaleFactor=selected_config.scale_factor,
            nlevels=selected_config.n_levels,
            edgeThreshold=selected_config.edge_threshold,
            firstLevel=selected_config.first_level,
            WTA_K=selected_config.wta_k,
            scoreType=selected_config.score_type,
            patchSize=selected_config.patch_size,
            fastThreshold=selected_config.fast_threshold,
        )

    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    if not keypoints or descriptors is None or descriptors.size == 0:
        return [], None
    return list(keypoints), descriptors

