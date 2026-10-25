"""Unit and Integration Tests for Phase 1: Workload & Grounded Evaluation Harness."""

import os
import sys
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig, EdgeDetectionResult
from src.eval.evaluator import GroundedEvaluator, QualityMetrics, TimingMetrics, EvaluationVerdict


class TestEdgeDetectorConfig:
    """Test validation rules for EdgeDetectorConfig."""

    def test_default_config_valid(self):
        config = EdgeDetector.get_baseline_config()
        assert config.low_threshold == 50
        assert config.high_threshold == 150
        assert config.aperture_size == 3
        assert config.blur_kernel == 5

    def test_high_threshold_must_exceed_low(self):
        with pytest.raises(ValueError, match="must be greater than"):
            EdgeDetectorConfig(low_threshold=100, high_threshold=80)

    def test_aperture_size_validation(self):
        with pytest.raises(ValueError, match="aperture_size must be 3, 5, or 7"):
            EdgeDetectorConfig(low_threshold=50, high_threshold=150, aperture_size=4)

    def test_blur_kernel_odd_requirement(self):
        with pytest.raises(ValueError, match="must be an odd integer"):
            EdgeDetectorConfig(low_threshold=50, high_threshold=150, blur_kernel=4)


class TestEdgeDetectorExecution:
    """Test EdgeDetector execution pipeline."""

    def test_execute_on_synthetic_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)

        edges, result = EdgeDetector.execute(img)
        assert edges.shape == (100, 100)
        assert edges.dtype == np.uint8
        assert result.edge_pixel_count > 0
        assert result.status == "SUCCESS"
        assert result.execution_latency_ms > 0

    def test_oversized_image_rejected(self):
        oversized = np.zeros((1200, 1200), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds safe CPU limit"):
            EdgeDetector.execute(oversized)

    def test_empty_image_rejected(self):
        empty = np.array([], dtype=np.uint8)
        with pytest.raises(ValueError):
            EdgeDetector.execute(empty)


class TestGroundedEvaluatorMetrics:
    """Test GroundedEvaluator precision, recall, F1, and proxy indicators."""

    def test_perfect_edge_match(self):
        # When candidate exactly matches reference, F1 should be 1.0
        ref = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ref, (20, 20), (80, 80), 255, 1)

        metrics = GroundedEvaluator.evaluate_edge_quality(candidate_edges=ref, reference_edges=ref)
        assert metrics.has_ground_truth is True
        assert metrics.f1_score == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_zero_detected_edges_returns_zero_f1(self):
        ref = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ref, (20, 20), (80, 80), 255, 1)
        cand = np.zeros((100, 100), dtype=np.uint8)

        metrics = GroundedEvaluator.evaluate_edge_quality(candidate_edges=cand, reference_edges=ref)
        assert metrics.f1_score == 0.0
        assert metrics.recall == 0.0

    def test_proxy_metrics_computation(self):
        # Unlabelled image: test proxy metrics calculation
        cand = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(cand, (10, 10), (90, 10), 255, 1)

        metrics = GroundedEvaluator.evaluate_edge_quality(candidate_edges=cand, reference_edges=None)
        assert metrics.has_ground_truth is False
        assert metrics.is_proxy_metric is True
        assert 0.0 <= metrics.proxy_score <= 1.0
        assert metrics.edge_density > 0


class TestQualityGuardrails:
    """Test candidate evaluation and guardrail rejection."""

    def test_guardrail_rejects_sub_threshold_quality(self):
        # Subtle contrast (intensity 60 on 0 background)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (60, 60, 60), -1)
        ref = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ref, (20, 20), (80, 80), 255, 1)

        # High thresholds (240-250) will miss all edges with gradient ~60
        bad_config = EdgeDetectorConfig(low_threshold=240, high_threshold=250, blur_kernel=1)
        edges, verdict = GroundedEvaluator.evaluate_candidate(
            image=img,
            candidate_config=bad_config,
            reference_edges=ref,
            min_quality_threshold=0.75
        )

        assert verdict.is_acceptable is False
        assert verdict.status == "REJECTED_QUALITY"
        assert len(verdict.rejection_reasons) > 0

    def test_speedup_calculation_vs_baseline(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)

        fast_config = EdgeDetectorConfig(low_threshold=50, high_threshold=150, blur_kernel=1)
        mock_baseline_timing = TimingMetrics(
            median_latency_ms=10.0,
            mean_latency_ms=10.0,
            min_latency_ms=9.5,
            max_latency_ms=10.5,
            std_latency_ms=0.3
        )

        edges, verdict = GroundedEvaluator.evaluate_candidate(
            image=img,
            candidate_config=fast_config,
            baseline_timing=mock_baseline_timing
        )

        assert verdict.speedup_factor is not None
        assert verdict.speedup_factor > 1.0  # Should be faster than 10.0 ms mock baseline


class TestCuratedDatasetIntegrity:
    """Verify that all 5 benchmark images and ground truth references exist and load."""

    def test_all_manifest_images_exist(self):
        import json
        manifest_path = "data/dataset_manifest.json"
        assert os.path.exists(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) >= 5
        for item in manifest:
            assert os.path.exists(item["image_path"]), f"Missing image: {item['image_path']}"
            assert os.path.exists(item["reference_path"]), f"Missing reference: {item['reference_path']}"
            img = cv2.imread(item["image_path"])
            ref = cv2.imread(item["reference_path"], cv2.IMREAD_GRAYSCALE)
            assert img is not None
            assert ref is not None
            assert img.shape[:2] == ref.shape[:2]
