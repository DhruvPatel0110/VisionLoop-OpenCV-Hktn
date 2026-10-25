"""Executes and records standardized baseline benchmarks across all curated test images.

Produces:
1. data/baseline_outputs/{image_id}_baseline.png: Visual output of baseline run.
2. data/baseline_results.json: Structured benchmark reference file containing timing and quality scores.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import cv2
import numpy as np

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig
from src.eval.evaluator import GroundedEvaluator


def run_all_baselines():
    manifest_path = "data/dataset_manifest.json"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run data/create_sample_dataset.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir = "data/baseline_outputs"
    os.makedirs(output_dir, exist_ok=True)

    baseline_config = EdgeDetector.get_baseline_config()
    print("=" * 80)
    print("VISIONLOOP: RECORDING BASELINE BENCHMARKS")
    print(f"Standard Baseline Config: {baseline_config.model_dump()}")
    print("=" * 80)

    results = []

    for item in manifest:
        img_id = item["id"]
        img_path = item["image_path"]
        ref_path = item.get("reference_path")

        print(f"\nProcessing [{img_id}] - {item['name']}...")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE) if ref_path and os.path.exists(ref_path) else None

        # Full evaluation run
        edges, verdict = GroundedEvaluator.evaluate_candidate(
            image=image,
            candidate_config=baseline_config,
            reference_edges=reference
        )

        # Save visual output
        out_path = os.path.join(output_dir, f"{img_id}_baseline.png").replace("\\", "/")
        cv2.imwrite(out_path, edges)

        record = {
            "image_id": img_id,
            "image_name": item["name"],
            "category": item["category"],
            "image_path": img_path,
            "baseline_output_path": out_path,
            "dimensions": item["dimensions"],
            "config": baseline_config.model_dump(),
            "timing": verdict.timing_metrics.model_dump(),
            "quality": verdict.quality_metrics.model_dump(),
            "verdict": {
                "status": verdict.status,
                "is_acceptable": verdict.is_acceptable,
                "primary_score": verdict.primary_quality_score,
                "metric_type": verdict.quality_metric_type
            }
        }
        results.append(record)

        print(f"  -> Median Latency: {verdict.timing_metrics.median_latency_ms:.2f} ms (std: {verdict.timing_metrics.std_latency_ms:.3f} ms)")
        if verdict.quality_metrics.has_ground_truth:
            print(f"  -> Ground Truth F1: {verdict.quality_metrics.f1_score:.4f} (Precision: {verdict.quality_metrics.precision:.4f}, Recall: {verdict.quality_metrics.recall:.4f})")
        else:
            print(f"  -> Proxy Score: {verdict.quality_metrics.proxy_score:.4f} (Continuity: {verdict.quality_metrics.edge_continuity:.3f}, Noise: {verdict.quality_metrics.noise_ratio:.3f})")
        print(f"  -> Saved output to: {out_path}")

    # Save results json
    results_path = "data/baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_images": len(results),
            "baseline_config": baseline_config.model_dump(),
            "results": results
        }, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BASELINE BENCHMARKS COMPLETE: {len(results)} images evaluated.")
    print(f"Results saved to: {results_path}")
    print("=" * 80)
    return results_path


if __name__ == "__main__":
    run_all_baselines()
