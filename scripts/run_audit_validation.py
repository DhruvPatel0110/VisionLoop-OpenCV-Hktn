"""Audit verification script for VisionLoop experiment pipeline.

Runs real OpenCV 5 workloads across:
1. Geometric shapes without reference (Unlabelled custom upload - Proxy evaluation)
2. Geometric shapes with verified ground-truth reference (F1 tolerance evaluation)
3. Low-contrast benchmark image with verified ground-truth reference (Failure recovery & tuning)
"""

import os
import sys
import shutil
import cv2

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.coordinator import VisionLoopCoordinator


def audit_run(image_path: str, ref_path: str, image_id: str, min_q: float = 0.70):
    print("=" * 85)
    print(f"AUDIT RUN: {image_id}")
    print(f"Image Source:     {image_path}")
    print(f"Reference Source: {ref_path if ref_path else 'None (Unlabelled - Proxy Quality Metrics Active)'}")
    print(f"Quality Target:   {min_q:.2f}")
    print("=" * 85)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE) if ref_path else None

    output_dir = "runs_audit"
    os.makedirs(output_dir, exist_ok=True)

    coord = VisionLoopCoordinator(max_iterations=5, target_min_quality=min_q, output_dir=output_dir)
    summary = coord.run(img, reference_edges=ref, image_id=image_id, save_candidate_images=True)

    print("\n" + "-" * 85)
    print(f"SUMMARY FOR {image_id}:")
    print(f"Session ID:         {summary.session_id}")
    print(f"Evaluation Mode:    {summary.baseline_quality_type}")
    print(f"Has Ground Truth:   {summary.has_ground_truth}")
    print(f"Baseline:           {summary.baseline_latency_ms:.2f} ms | Score: {summary.baseline_quality_score:.4f}")
    if summary.best_latency_ms:
        print(f"Best Incumbent:     {summary.best_latency_ms:.2f} ms | Score: {summary.best_quality_score:.4f} | Speedup: {summary.speedup_factor:.2f}x")
    print("-" * 85)

    header = f"{'#':<3} | {'Parameters (Canny)':<30} | {'Latency':<10} | {'Quality Metric':<18} | {'Delta Latency':<14} | {'Status':<16} | {'Winner?'}"
    print(header)
    print("-" * len(header))

    for t in summary.traces:
        cfg = t.config
        cfg_str = f"({cfg.low_threshold}, {cfg.high_threshold}), k={cfg.aperture_size}, blur={cfg.blur_kernel}"
        lat_str = f"{t.verdict.timing_metrics.median_latency_ms:.2f} ms"
        score_val = t.verdict.primary_quality_score
        score_str = f"{score_val:.4f} (F1)" if summary.has_ground_truth else f"{score_val:.4f} (Proxy)"
        lat_red = t.verdict.latency_reduction_pct
        delta_str = f"{lat_red:+.1f}%" if lat_red is not None else "Baseline"
        winner_str = "SELECTED" if t.is_incumbent_best else "-"
        print(f"{t.iteration:<3} | {cfg_str:<30} | {lat_str:<10} | {score_str:<18} | {delta_str:<14} | {t.verdict.status:<16} | {winner_str}")
    print()
    return summary


if __name__ == "__main__":
    # Test 1: Uploaded geometric image without reference (Unlabelled - Proxy evaluation)
    audit_run("data/test_images/sample_01.png", None, "geometric_shapes_unlabelled", min_q=0.70)

    # Test 2: Sample 01 with genuine ground-truth reference
    audit_run("data/test_images/sample_01.png", "data/reference_edges/sample_01_ref.png", "sample_01_ground_truth", min_q=0.20)

    # Test 3: Sample 04 low-contrast with genuine ground-truth reference
    audit_run("data/test_images/sample_04_low_contrast.png", "data/reference_edges/sample_04_low_contrast_ref.png", "sample_04_low_contrast_ground_truth", min_q=0.70)

    # Clean up runs_audit
    shutil.rmtree("runs_audit", ignore_errors=True)
