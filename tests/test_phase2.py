"""Unit and Integration Tests for Phase 2: Autonomous Agent Loop & Adaptive Decision Engine."""

import os
import sys
import json
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig
from src.eval.evaluator import GroundedEvaluator, EvaluationVerdict, QualityMetrics, TimingMetrics
from src.agent.optimizer import AdaptiveOptimizer, AdaptiveDecision
from src.agent.coordinator import VisionLoopCoordinator, OptimizationSummary, AgentState


class TestAdaptiveOptimizer:
    """Test perception-driven decision logic in AdaptiveOptimizer."""

    def test_recovers_from_edge_silence(self):
        """When zero edges are detected, optimizer must aggressively lower thresholds."""
        cfg = EdgeDetectorConfig(low_threshold=50, high_threshold=150, blur_kernel=5)
        # Mock verdict with zero edges (density = 0)
        qm = QualityMetrics(
            has_ground_truth=True,
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            proxy_score=0.1,
            edge_density=0.0,
            edge_continuity=0.0,
            noise_ratio=0.0
        )
        tm = TimingMetrics(median_latency_ms=2.0, mean_latency_ms=2.0, min_latency_ms=1.9, max_latency_ms=2.1, std_latency_ms=0.1)
        verdict = EvaluationVerdict(
            status="REJECTED_QUALITY",
            is_acceptable=False,
            primary_quality_score=0.0,
            quality_metric_type="GROUND_TRUTH_F1",
            quality_guardrail_threshold=0.70,
            rejection_reasons=["Zero edges detected"],
            quality_metrics=qm,
            timing_metrics=tm,
            config=cfg
        )

        decision = AdaptiveOptimizer.propose_next_configuration(
            last_verdict=verdict,
            best_verdict=None,
            tested_configs=[cfg],
            iteration=1,
            target_min_quality=0.70
        )

        assert decision.action_type == "RECOVER_SILENT_EDGES"
        assert decision.proposed_config.low_threshold < cfg.low_threshold
        assert decision.proposed_config.high_threshold < cfg.high_threshold
        assert "zero significant edges" in decision.rationale

    def test_suppresses_excessive_noise(self):
        """When noise artifacts explode, optimizer must raise threshold and increase blur."""
        cfg = EdgeDetectorConfig(low_threshold=40, high_threshold=100, blur_kernel=1)
        qm = QualityMetrics(
            has_ground_truth=True,
            f1_score=0.25,
            precision=0.15,  # Very low precision due to noise
            recall=0.85,
            proxy_score=0.3,
            edge_density=0.15,
            edge_continuity=0.2,
            noise_ratio=0.35  # High noise ratio
        )
        tm = TimingMetrics(median_latency_ms=1.5, mean_latency_ms=1.5, min_latency_ms=1.4, max_latency_ms=1.6, std_latency_ms=0.1)
        verdict = EvaluationVerdict(
            status="REJECTED_QUALITY",
            is_acceptable=False,
            primary_quality_score=0.25,
            quality_metric_type="GROUND_TRUTH_F1",
            quality_guardrail_threshold=0.70,
            rejection_reasons=["Excessive noise"],
            quality_metrics=qm,
            timing_metrics=tm,
            config=cfg
        )

        decision = AdaptiveOptimizer.propose_next_configuration(
            last_verdict=verdict,
            best_verdict=None,
            tested_configs=[cfg],
            iteration=1,
            target_min_quality=0.70
        )

        assert decision.action_type == "SUPPRESS_NOISE"
        assert decision.proposed_config.low_threshold > cfg.low_threshold
        assert decision.proposed_config.blur_kernel >= 3
        assert "Noise" in decision.diagnosis or "noisy" in decision.rationale

    def test_acceptable_candidate_triggers_speedup_attempt(self):
        """When candidate meets quality threshold, optimizer attempts latency optimization."""
        cfg = EdgeDetectorConfig(low_threshold=50, high_threshold=150, blur_kernel=5)
        qm = QualityMetrics(
            has_ground_truth=True,
            f1_score=0.88,
            precision=0.90,
            recall=0.86,
            proxy_score=0.85,
            edge_density=0.03,
            edge_continuity=0.8,
            noise_ratio=0.02
        )
        tm = TimingMetrics(median_latency_ms=2.5, mean_latency_ms=2.5, min_latency_ms=2.4, max_latency_ms=2.6, std_latency_ms=0.1)
        verdict = EvaluationVerdict(
            status="ACCEPTED",
            is_acceptable=True,
            primary_quality_score=0.88,
            quality_metric_type="GROUND_TRUTH_F1",
            quality_guardrail_threshold=0.70,
            quality_metrics=qm,
            timing_metrics=tm,
            config=cfg
        )

        decision = AdaptiveOptimizer.propose_next_configuration(
            last_verdict=verdict,
            best_verdict=verdict,
            tested_configs=[cfg],
            iteration=1,
            target_min_quality=0.70
        )

        assert decision.action_type in ("SPEEDUP_REDUCE_BLUR", "SPEEDUP_L1_GRADIENT")
        assert "Quality guardrail satisfied" in decision.diagnosis


class TestVisionLoopCoordinator:
    """Test full execution of the autonomous loop coordinator."""

    def test_coordinator_runs_end_to_end(self, tmp_path):
        # Create small synthetic image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (240, 240, 240), -1)
        ref = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ref, (20, 20), (80, 80), 255, 1)

        coordinator = VisionLoopCoordinator(
            max_iterations=3,
            max_seconds=10.0,
            target_min_quality=0.60,
            output_dir=str(tmp_path)
        )

        summary = coordinator.run(
            image=img,
            reference_edges=ref,
            image_id="test_synthetic"
        )

        assert summary.state == AgentState.COMPLETED
        assert summary.iterations_completed == 3
        assert len(summary.traces) == 4  # Iteration 0 (baseline) + 3 candidates
        assert os.path.exists(summary.trace_json_path)
        assert os.path.exists(summary.trace_markdown_path)

        # Verify JSON trace readability
        with open(summary.trace_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["session_id"] == summary.session_id
            assert "traces" in data
            assert len(data["traces"]) == 4

    def test_coordinator_recovers_failing_baseline(self, tmp_path):
        """Verify that agent recovers an image where baseline scored 0.0."""
        # Low contrast image (delta ~30) where baseline thresholds (50, 150) fail completely
        img = np.ones((100, 100, 3), dtype=np.uint8) * 40
        cv2.rectangle(img, (25, 25), (75, 75), (70, 70, 70), -1)
        ref = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(ref, (25, 25), (75, 75), 255, 1)

        coordinator = VisionLoopCoordinator(
            max_iterations=4,
            max_seconds=10.0,
            target_min_quality=0.70,
            output_dir=str(tmp_path)
        )

        summary = coordinator.run(
            image=img,
            reference_edges=ref,
            image_id="low_contrast_test"
        )

        assert summary.baseline_quality_score == 0.0  # Baseline should fail
        assert summary.best_quality_score is not None
        assert summary.best_quality_score >= 0.70    # Agent must recover quality
        assert summary.best_config.low_threshold < 50
