"""Grounded Quality Evaluation Engine and Benchmark Harness for OpenCV 5 Workloads.

Implements:
1. Ground-Truth Quality Metrics: Precision, Recall, and F1-score against reference edge maps
   (using standard 1-pixel morphological tolerance).
2. Proxy Quality Metrics (for unlabelled images): Edge continuity, density, and noise penalty.
3. Multi-Pass Timing Harness: Warmup passes + median of repeated runs to filter CPU jitter.
4. Quality Guardrails: Strict enforcement of minimum quality thresholds with explicit rejection reasons.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import cv2
import numpy as np
from pydantic import BaseModel, Field

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig, EdgeDetectionResult


class QualityMetrics(BaseModel):
    """Detailed quality metrics computed for an edge detection output."""
    has_ground_truth: bool = Field(description="True if scored against ground-truth reference")
    f1_score: Optional[float] = Field(default=None, description="Dice/F1 harmonic mean of precision and recall")
    precision: Optional[float] = Field(default=None, description="Fraction of detected edge pixels matching reference")
    recall: Optional[float] = Field(default=None, description="Fraction of reference edge pixels successfully detected")
    tolerance_pixels: int = Field(default=1, description="Pixel tolerance distance for matching edges")

    # Proxy metrics (always computed, explicitly tagged)
    is_proxy_metric: bool = Field(default=False, description="True if composite score is purely proxy-based")
    proxy_score: float = Field(description="Normalized composite proxy quality score [0.0 - 1.0]")
    edge_density: float = Field(description="Fraction of image pixels that are edges")
    edge_continuity: float = Field(description="Metric of edge continuity vs fragmentation [0.0 - 1.0]")
    noise_ratio: float = Field(description="Fraction of detected edge pixels that are isolated single pixels")


class TimingMetrics(BaseModel):
    """High-precision execution latency statistics."""
    median_latency_ms: float = Field(description="Median execution time across timed runs in ms")
    mean_latency_ms: float = Field(description="Mean execution time in ms")
    min_latency_ms: float = Field(description="Fastest execution time in ms")
    max_latency_ms: float = Field(description="Slowest execution time in ms")
    std_latency_ms: float = Field(description="Standard deviation of execution time in ms")
    timed_iterations: int = Field(default=5, description="Number of timed evaluation runs")
    warmup_iterations: int = Field(default=3, description="Number of untimed warmup runs")


class EvaluationVerdict(BaseModel):
    """Overall verdict and guardrail evaluation for a candidate configuration."""
    status: str = Field(description="Verdict: 'ACCEPTED', 'REJECTED_QUALITY', 'REJECTED_LATENCY', or 'ERROR'")
    is_acceptable: bool = Field(description="True if candidate satisfies all quality guardrails")
    primary_quality_score: float = Field(description="Primary quality score (F1 if GT available, else proxy)")
    quality_metric_type: str = Field(description="'GROUND_TRUTH_F1' or 'PROXY_COMPOSITE'")
    speedup_factor: Optional[float] = Field(default=None, description="T_baseline / T_candidate (e.g. 1.25 = 25% faster)")
    latency_reduction_pct: Optional[float] = Field(default=None, description="Percentage latency reduction vs baseline")
    quality_guardrail_threshold: float = Field(description="Minimum acceptable score required to pass")
    rejection_reasons: List[str] = Field(default_factory=list, description="Explicit reasons for rejection")
    quality_metrics: QualityMetrics
    timing_metrics: TimingMetrics
    config: EdgeDetectorConfig


class GroundedEvaluator:
    """Evaluates edge detection outputs with rigorous ground-truth metrics and noise-resilient timing."""

    DEFAULT_MIN_F1_THRESHOLD: float = 0.70
    DEFAULT_MIN_PROXY_THRESHOLD: float = 0.60

    @classmethod
    def evaluate_edge_quality(
        cls,
        candidate_edges: np.ndarray,
        reference_edges: Optional[np.ndarray] = None,
        tolerance: int = 1
    ) -> QualityMetrics:
        """
        Computes precision, recall, F1 against reference edge map, and proxy quality indicators.
        """
        if candidate_edges is None or candidate_edges.size == 0:
            raise ValueError("Candidate edges array is empty or None")

        h, w = candidate_edges.shape[:2]
        total_pixels = h * w
        cand_bin = (candidate_edges > 0).astype(np.uint8)
        cand_count = int(np.count_nonzero(cand_bin))
        edge_density = cand_count / total_pixels if total_pixels > 0 else 0.0

        # --- Calculate Proxy Indicators ---
        # 1. Noise ratio: isolated edge pixels (no 8-connected neighbors)
        kernel_8 = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(cand_bin, -1, kernel_8) - cand_bin
        isolated_pixels = int(np.count_nonzero((cand_bin > 0) & (neighbor_count == 0)))
        noise_ratio = isolated_pixels / cand_count if cand_count > 0 else 0.0

        # 2. Edge continuity: ratio of edge pixels to connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand_bin, connectivity=8)
        num_components = max(1, num_labels - 1)  # subtract background
        avg_component_size = cand_count / num_components if num_components > 0 else 0.0
        # Normalize continuity score (longer continuous lines score higher, fragmented dust scores lower)
        continuity_score = float(np.clip(avg_component_size / 25.0, 0.0, 1.0))

        # 3. Composite Proxy Score [0.0 - 1.0]
        # Rewarding healthy edge density (0.5% - 10%), high continuity, and low noise
        density_fitness = float(np.clip(1.0 - abs(edge_density - 0.03) * 15.0, 0.1, 1.0))
        noise_penalty = max(0.0, 1.0 - (noise_ratio * 4.0))
        proxy_score = round(float(0.4 * continuity_score + 0.35 * noise_penalty + 0.25 * density_fitness), 4)

        # --- Calculate Ground Truth Metrics (if reference available) ---
        has_gt = reference_edges is not None
        f1_score = None
        precision = None
        recall = None

        if has_gt:
            if reference_edges.shape[:2] != (h, w):
                raise ValueError(f"Reference shape {reference_edges.shape} does not match candidate {candidate_edges.shape}")

            ref_bin = (reference_edges > 0).astype(np.uint8)
            ref_count = int(np.count_nonzero(ref_bin))

            if ref_count == 0 and cand_count == 0:
                f1_score, precision, recall = 1.0, 1.0, 1.0
            elif ref_count == 0 or cand_count == 0:
                f1_score, precision, recall = 0.0, 0.0, 0.0
            else:
                # Morphological dilation by tolerance distance (standard BSDS evaluation practice)
                if tolerance > 0:
                    kernel_tol = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * tolerance + 1, 2 * tolerance + 1))
                    ref_dilated = cv2.dilate(ref_bin, kernel_tol)
                    cand_dilated = cv2.dilate(cand_bin, kernel_tol)

                    # True Positives: candidate edges within tolerance of reference
                    tp_precision = np.count_nonzero((cand_bin > 0) & (ref_dilated > 0))
                    precision = tp_precision / cand_count if cand_count > 0 else 0.0

                    # True Positives: reference edges detected by candidate within tolerance
                    tp_recall = np.count_nonzero((ref_bin > 0) & (cand_dilated > 0))
                    recall = tp_recall / ref_count if ref_count > 0 else 0.0
                else:
                    tp = np.count_nonzero((cand_bin > 0) & (ref_bin > 0))
                    precision = tp / cand_count if cand_count > 0 else 0.0
                    recall = tp / ref_count if ref_count > 0 else 0.0

                if (precision + recall) > 0:
                    f1_score = 2.0 * (precision * recall) / (precision + recall)
                else:
                    f1_score = 0.0

                f1_score = round(float(f1_score), 4)
                precision = round(float(precision), 4)
                recall = round(float(recall), 4)

        return QualityMetrics(
            has_ground_truth=has_gt,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
            tolerance_pixels=tolerance,
            is_proxy_metric=not has_gt,
            proxy_score=proxy_score,
            edge_density=round(edge_density, 5),
            edge_continuity=round(continuity_score, 4),
            noise_ratio=round(noise_ratio, 4)
        )

    @classmethod
    def benchmark_timing(
        cls,
        image: np.ndarray,
        config: EdgeDetectorConfig,
        warmup_iterations: int = 3,
        timed_iterations: int = 5
    ) -> Tuple[np.ndarray, TimingMetrics]:
        """
        Executes multi-pass timing to filter out CPU noise and thermal fluctuations.
        """
        # Warmup passes
        edges = None
        for _ in range(warmup_iterations):
            edges, _ = EdgeDetector.execute(image, config)

        # Timed passes
        latencies_ms: List[float] = []
        for _ in range(timed_iterations):
            t0 = time.perf_counter_ns()
            edges, _ = EdgeDetector.execute(image, config)
            t_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies_ms.append(t_ms)

        med_lat = float(np.median(latencies_ms))
        mean_lat = float(np.mean(latencies_ms))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        std_lat = float(np.std(latencies_ms))

        timing = TimingMetrics(
            median_latency_ms=round(med_lat, 3),
            mean_latency_ms=round(mean_lat, 3),
            min_latency_ms=round(min_lat, 3),
            max_latency_ms=round(max_lat, 3),
            std_latency_ms=round(std_lat, 4),
            timed_iterations=timed_iterations,
            warmup_iterations=warmup_iterations
        )

        return edges, timing

    @classmethod
    def evaluate_candidate(
        cls,
        image: np.ndarray,
        candidate_config: EdgeDetectorConfig,
        reference_edges: Optional[np.ndarray] = None,
        baseline_timing: Optional[TimingMetrics] = None,
        baseline_is_acceptable: bool = True,
        min_quality_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, EvaluationVerdict]:
        """
        Full evaluation of a candidate configuration against quality guardrails and baseline timing.
        """
        # 1. Benchmark candidate runtime
        edges, timing = cls.benchmark_timing(image, candidate_config)

        # 2. Compute quality metrics
        quality = cls.evaluate_edge_quality(edges, reference_edges=reference_edges)

        # 3. Determine primary score & threshold
        if quality.has_ground_truth and quality.f1_score is not None:
            primary_score = quality.f1_score
            metric_type = "GROUND_TRUTH_F1"
            threshold = min_quality_threshold or cls.DEFAULT_MIN_F1_THRESHOLD
        else:
            primary_score = quality.proxy_score
            metric_type = "PROXY_COMPOSITE"
            threshold = min_quality_threshold or cls.DEFAULT_MIN_PROXY_THRESHOLD

        # 4. Compute speedup relative to baseline if provided
        speedup = None
        latency_reduction_pct = None
        if baseline_timing is not None and baseline_timing.median_latency_ms > 0:
            speedup = round(baseline_timing.median_latency_ms / max(0.001, timing.median_latency_ms), 3)
            red_pct = ((baseline_timing.median_latency_ms - timing.median_latency_ms) / baseline_timing.median_latency_ms) * 100.0
            latency_reduction_pct = round(red_pct, 2)

        # 5. Apply Quality Guardrails
        rejection_reasons: List[str] = []
        is_acceptable = True

        if primary_score < threshold:
            is_acceptable = False
            rejection_reasons.append(
                f"Quality score {primary_score:.3f} failed minimum guardrail threshold of {threshold:.3f}"
            )

        # Secondary guardrails: noise blowout or zero edges
        if quality.edge_density < 0.0005:
            is_acceptable = False
            rejection_reasons.append("Severe edge suppression: Under 0.05% edge density detected")

        if quality.noise_ratio > 0.40:
            is_acceptable = False
            rejection_reasons.append(f"Excessive noise: {quality.noise_ratio*100:.1f}% of edge pixels are isolated noise")

        # Determine verdict status
        if not is_acceptable:
            status = "REJECTED_QUALITY"
        elif baseline_is_acceptable and speedup is not None and speedup < 0.90:
            # Only enforce latency rejection if baseline itself was an acceptable quality output
            status = "REJECTED_LATENCY"
            rejection_reasons.append(f"Candidate is slower than baseline (speedup {speedup}x < 0.90x)")
            is_acceptable = False
        else:
            status = "ACCEPTED"

        verdict = EvaluationVerdict(
            status=status,
            is_acceptable=is_acceptable,
            primary_quality_score=primary_score,
            quality_metric_type=metric_type,
            speedup_factor=speedup,
            latency_reduction_pct=latency_reduction_pct,
            quality_guardrail_threshold=threshold,
            rejection_reasons=rejection_reasons,
            quality_metrics=quality,
            timing_metrics=timing,
            config=candidate_config
        )

        return edges, verdict
