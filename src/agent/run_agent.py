"""CLI Entrypoint to run the VisionLoop autonomous agent on any test image.

Usage:
    python src/agent/run_agent.py --image data/test_images/sample_04_low_contrast.png
    python src/agent/run_agent.py --image data/test_images/sample_03_noisy.png --max-iter 5
"""

import os
import sys
import argparse
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agent.coordinator import VisionLoopCoordinator


def main():
    parser = argparse.ArgumentParser(description="Run VisionLoop Autonomous Optimization Agent")
    parser.add_argument("--image", type=str, default="data/test_images/sample_04_low_contrast.png", help="Path to input image")
    parser.add_argument("--reference", type=str, default=None, help="Optional path to reference edge map")
    parser.add_argument("--max-iter", type=int, default=5, help="Maximum number of candidate iterations")
    parser.add_argument("--timeout", type=float, default=25.0, help="Maximum execution time budget in seconds")
    parser.add_argument("--min-quality", type=float, default=0.70, help="Minimum quality score guardrail")
    parser.add_argument("--output-dir", type=str, default="runs", help="Directory to save traces and reports")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Failed to decode image: {args.image}")
        sys.exit(1)

    # Auto-detect reference image if not specified
    ref_img = None
    if args.reference and os.path.exists(args.reference):
        ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    else:
        # Check standard reference naming convention
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        potential_ref = os.path.join("data/reference_edges", f"{base_name}_ref.png")
        if os.path.exists(potential_ref):
            print(f"Auto-detected ground truth reference: {potential_ref}")
            ref_img = cv2.imread(potential_ref, cv2.IMREAD_GRAYSCALE)

    image_id = os.path.splitext(os.path.basename(args.image))[0]

    coordinator = VisionLoopCoordinator(
        max_iterations=args.max_iter,
        max_seconds=args.timeout,
        target_min_quality=args.min_quality,
        output_dir=args.output_dir
    )

    summary = coordinator.run(
        image=img,
        reference_edges=ref_img,
        image_id=image_id
    )

    print(f"\nOptimization finished with status: {summary.state}")
    print(f"JSON Trace: {summary.trace_json_path}")
    print(f"Markdown Report: {summary.trace_markdown_path}")


if __name__ == "__main__":
    main()
