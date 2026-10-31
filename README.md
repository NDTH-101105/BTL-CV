# Panorama Image Stitching

This project implements the Computer Vision course assignment for panoramic photo stitching. It builds a wider view from overlapping images using a feature-based pipeline implemented with Python and OpenCV. The same loading, matching, homography, warping, and blending stages are used for SIFT and ORB; the detector/descriptor and matching distance differ by method.

## Project layout

```text
BTL-CV/
├── data/
│   └── boat/                  # Ordered input sequence: 01.jpg, 02.jpg, ...
├── outputs/
│   ├── sift/                  # SIFT visualizations, warps, panorama, CSV
│   ├── orb/                   # ORB visualizations, warps, panorama, CSV
│   └── evaluation.csv         # Combined SIFT/ORB measurements
├── src/
│   ├── config.py              # Paths and shared SIFT/ORB/matching/RANSAC settings
│   ├── image_io.py            # Image loading and grayscale preprocessing
│   ├── features.py            # SIFT and ORB feature extraction
│   ├── matching.py            # BFMatcher KNN matching and ratio test
│   ├── homography.py          # RANSAC homography and inlier visualization
│   ├── warping.py             # Transform composition, canvas, image/mask warping
│   ├── blending.py            # Mask-aware feather blending and alpha PNG output
│   ├── pipeline.py            # Orchestration for one feature method
│   ├── evaluation.py          # Metrics data and CSV export
│   └── main.py                # Command-line entry point
└── tests/                     # Unit and integration tests
```

`outputs/` is ignored by Git, so generated images and CSV files remain local unless that ignore rule is deliberately changed.

## Environment

The project was run on Windows with Python 3.13.2, OpenCV 5.0.0, NumPy 2.3.5, and pytest 8.4.2. In the current VS Code setup, the tested interpreter is:

```text
C:\Users\Admin\anaconda3\python.exe
```

In VS Code, run **Python: Select Interpreter** and select that interpreter (or the interpreter where you installed the dependencies). From PowerShell at the project root, install the packages into the selected interpreter:

```powershell
python -m pip install opencv-python numpy pytest
```

Confirm the interpreter and imports if needed:

```powershell
python -c "import sys, cv2, numpy; print(sys.executable); print(cv2.__version__); print(numpy.__version__)"
```

## Input images

Place one ordered sequence of at least two overlapping color images in `data/boat/`. The default sample sequence contains:

```text
data/boat/01.jpg
data/boat/02.jpg
data/boat/03.jpg
data/boat/04.jpg
```

The loader sorts supported image files case-insensitively by filename. Use names such as `01.jpg`, `02.jpg`, etc. in capture order. The current example images are 3888×2592; the loader does not resize them. For useful homographies, images should show the same scene with substantial overlap and should ideally be captured by rotating the camera around a similar viewpoint.

## Run the project

Run both methods on the default `data/boat/` directory:

```powershell
python -m src.main
```

Run one method only:

```powershell
python -m src.main --methods sift
python -m src.main --methods orb
```

Override the data/output folders or reference image index if needed:

```powershell
python -m src.main --data-dir .\data\boat --output-dir .\outputs --reference-index 1
```

By default, the middle image in filename order is the reference. For four images, this is `02.jpg`. The pipeline matches adjacent pairs (`01–02`, `02–03`, `03–04`), composes their source-to-destination homographies into the reference coordinate system, warps each image and mask, then blends valid pixels. A matching or homography failure identifies the method and image pair and prevents that method from producing a misleading panorama.

## Outputs

Each method writes to its own folder:

- `matches_*.jpg`: ratio-filtered feature matches for each adjacent pair.
- `homography_*.jpg`: inlier/outlier correspondences (green/red).
- `warped_*_image.png` and `warped_*_mask.png`: each input warped to the shared canvas.
- `panorama_before_blending.png`: first-valid-pixel composite for seam comparison.
- `panorama_blended.png`: feather-blended BGR panorama on the full rectangular canvas.
- `panorama_transparent.png`: same panorama with invalid canvas pixels transparent (alpha from the valid mask).
- `panorama_valid_mask.png` and `panorama_overlap_mask.png`: valid-pixel and overlap masks.
- `evaluation.csv`: image-level keypoints, pair-level matching/homography metrics, and stage timings.

`outputs/evaluation.csv` combines SIFT and ORB records. CSV records are typed as `image`, `pair`, `summary`, or `failure`; columns that do not apply to a record type are intentionally blank. Times are in seconds, inlier ratio is inliers divided by good matches, and reprojection error is the mean Euclidean inlier error in pixels. The CSV contains measurements, not subjective visual ratings.

## Tests

Run the full test suite:

```powershell
python -m pytest -q tests
```

The integration tests use the four-image boat sample and write artifacts to pytest temporary directories.

## Current limitations

- The rectangular panorama canvas includes areas outside the warped image masks. Use `panorama_transparent.png` where a transparent background is supported. A rectangular crop that removes every invalid corner would discard some valid image content for this sample.
- Feather blending reduces hard transitions but cannot correct parallax, moving objects, or geometric misalignment; these can still appear as ghosting. Seam optimization and exposure compensation are not implemented.
- Pairwise homographies are chained without bundle adjustment, so small errors can accumulate along longer sequences.
- The implementation is intended for a short ordered sequence with overlap; it does not automatically choose image order or detect disconnected groups.
- SIFT produces many descriptors with the current default configuration. On the sample set, feather blending and writing large warped canvases dominate total runtime.

