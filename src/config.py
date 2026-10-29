"""Project paths and shared settings for the image stitching pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# The project root is the parent of ``src``. Resolving it here makes the
# defaults independent of the directory from which Python is launched.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class SIFTConfig:
    """Parameters passed to OpenCV's SIFT detector and descriptor."""

    nfeatures: int = 0
    n_octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    sigma: float = 1.6


@dataclass(frozen=True)
class ORBConfig:
    """Parameters passed to OpenCV's ORB detector and descriptor."""

    nfeatures: int = 2500
    scale_factor: float = 1.2
    n_levels: int = 8
    edge_threshold: int = 31
    first_level: int = 0
    wta_k: int = 2
    score_type: int = 0  # cv2.ORB_HARRIS_SCORE; kept numeric to avoid importing cv2 here.
    patch_size: int = 31
    fast_threshold: int = 20


@dataclass(frozen=True)
class MatchingConfig:
    """Shared feature matching settings; distance metric depends on descriptor."""

    ratio_threshold: float = 0.75
    minimum_matches: int = 8
    norm_name_sift: str = "NORM_L2"
    norm_name_orb: str = "NORM_HAMMING"


@dataclass(frozen=True)
class RansacConfig:
    """Settings and validity thresholds for homography estimation."""

    reprojection_threshold: float = 4.0
    max_iterations: int = 2000
    confidence: float = 0.995
    minimum_inliers: int = 8
    minimum_inlier_ratio: float = 0.2


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and algorithm settings shared by the project."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "boat")
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    sift: SIFTConfig = field(default_factory=SIFTConfig)
    orb: ORBConfig = field(default_factory=ORBConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ransac: RansacConfig = field(default_factory=RansacConfig)


DEFAULT_CONFIG = ProjectConfig()

