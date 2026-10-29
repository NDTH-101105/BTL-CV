"""Command-line entry point for running SIFT and ORB panorama pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import DEFAULT_CONFIG
from src.evaluation import PipelineFailure, PipelineResult, write_evaluation_csv
from src.pipeline import PipelineError, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for data/output paths and feature methods."""

    parser = argparse.ArgumentParser(description="Run the custom image stitching pipeline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_CONFIG.data_dir)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("sift", "orb"),
        default=("sift", "orb"),
        help="Feature methods to run; both are used by default.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="Reference image index (default: central image in filename order).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected methods, report pair failures, and write aggregate CSV."""

    args = build_parser().parse_args(argv)
    results: list[PipelineResult] = []
    failures: list[PipelineFailure] = []
    for method in args.methods:
        try:
            result = run_pipeline(
                method,
                image_dir=args.data_dir,
                output_root=args.output_dir,
                reference_index=args.reference_index,
            )
        except PipelineError as exc:
            failures.append(
                PipelineFailure(
                    method=exc.method,
                    stage=exc.stage,
                    reason=exc.reason,
                    source_image=exc.source_image,
                    destination_image=exc.destination_image,
                )
            )
            print(f"FAILED: {exc}")
            continue
        except Exception as exc:
            failures.append(
                PipelineFailure(method=method, stage="setup", reason=str(exc))
            )
            print(f"FAILED: {method.upper()} setup: {exc}")
            continue

        results.append(result)
        print(
            f"{method.upper()}: panorama={result.artifacts['panorama_blended']} "
            f"size={result.canvas_width}x{result.canvas_height}; "
            f"total={result.timings.total:.3f}s"
        )
        for image in result.image_metrics:
            print(f"  {image.image_name}: {image.keypoints} keypoints ({image.feature_seconds:.3f}s)")
        for pair in result.pair_metrics:
            print(
                f"  {pair.source_image}-{pair.destination_image}: "
                f"good={pair.good_matches}, inliers={pair.inliers} "
                f"({pair.inlier_ratio:.3f}), reprojection={pair.reprojection_error:.3f}px"
            )

    aggregate_csv = write_evaluation_csv(
        results, args.output_dir / "evaluation.csv", failures=failures
    )
    print(f"Evaluation CSV: {aggregate_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

