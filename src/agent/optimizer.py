"""VisionLoop Adaptive Optimizer - Perception-Driven Parameter Decision Engine.

Translates visual and runtime evaluation measurements into subsequent parameter decisions:
  Perception (F1, Precision, Recall, Noise, Latency)
    -> Diagnosis (Under-detection, Noise artifacts, Latency regression, Improvement)
    -> Action (Dynamically computed parameter adjustment + explicit rationale).
"""

from typing import List, Optional, Tuple, Dict, Any
import copy
from src.vision.edge_detector import EdgeDetectorConfig
from src.eval.evaluator import EvaluationVerdict, QualityMetrics, TimingMetrics


class AdaptiveDecision(BaseModel if False else object):
    """Container for the agent's reasoning at a given step."""
    def __init__(
        self,
        proposed_config: EdgeDetectorConfig,
        diagnosis: str,
        rationale: str,
        expected_outcome: str,
        action_type: str
    ):
        self.proposed_config = proposed_config
        self.diagnosis = diagnosis
        self.rationale = rationale
        self.expected_outcome = expected_outcome
        self.action_type = action_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_config": self.proposed_config.model_dump(),
            "diagnosis": self.diagnosis,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "action_type": self.action_type
        }


class AdaptiveOptimizer:
    """Perception-driven decision engine that chooses the next test based on previous results."""

    @classmethod
    def propose_next_configuration(
        cls,
        last_verdict: EvaluationVerdict,
        best_verdict: Optional[EvaluationVerdict],
        tested_configs: List[EdgeDetectorConfig],
        iteration: int,
        target_min_quality: float = 0.70
    ) -> AdaptiveDecision:
        """
        Dynamically analyzes visual metrics and runtime measurements to generate the next candidate.
        Ensures the decision directly depends on observed evidence, NOT a pre-fixed list.
        """
        last_cfg = last_verdict.config
        qm = last_verdict.quality_metrics
        tm = last_verdict.timing_metrics

        # Extract key perceptual indicators
        has_gt = qm.has_ground_truth
        primary_score = last_verdict.primary_quality_score
        recall = qm.recall if has_gt and qm.recall is not None else (qm.edge_density * 30.0)
        precision = qm.precision if has_gt and qm.precision is not None else (1.0 - qm.noise_ratio)
        noise_ratio = qm.noise_ratio
        density = qm.edge_density
        speedup = last_verdict.speedup_factor or 1.0

        # Create candidate working copy
        next_low = last_cfg.low_threshold
        next_high = last_cfg.high_threshold
        next_blur = last_cfg.blur_kernel
        next_sigma = last_cfg.blur_sigma
        next_l2 = last_cfg.l2_gradient
        next_aperture = last_cfg.aperture_size

        diagnosis = ""
        rationale = ""
        expected_outcome = ""
        action_type = ""

        # =========================================================================
        # RULE 1: Complete Edge Silence (zero or near-zero edges detected)
        # =========================================================================
        if density < 0.001 or (has_gt and recall < 0.05):
            diagnosis = "Severe edge silence detected: image has subtle contrast that current thresholds completely missed."
            next_low = max(15, int(next_low * 0.40))
            next_high = max(35, int(next_high * 0.35))
            if next_high <= next_low:
                next_high = next_low + 20
            next_blur = 3  # mild blur to preserve faint edges
            next_sigma = 0.8
            action_type = "RECOVER_SILENT_EDGES"
            rationale = (
                f"Previous thresholds ({last_cfg.low_threshold}, {last_cfg.high_threshold}) detected zero significant edges. "
                f"Aggressively lowering thresholds to ({next_low}, {next_high}) and reducing blur kernel to {next_blur} to capture low-contrast boundaries."
            )
            expected_outcome = "Detect previously invisible low-contrast boundaries."

        # =========================================================================
        # RULE 2: Over-Suppression / Poor Recall (True edges are missing)
        # =========================================================================
        elif primary_score < target_min_quality and (has_gt and recall < 0.65):
            diagnosis = f"Over-suppression detected: Recall is only {recall:.2f} (missing real object boundaries)."
            # Lower high threshold to connect more edge segments
            next_high = max(30, int(next_high * 0.70))
            next_low = max(15, min(next_high - 15, int(next_low * 0.75)))
            if next_blur > 3:
                next_blur = 3
                next_sigma = 0.9
            action_type = "BOOST_RECALL"
            rationale = (
                f"Quality score {primary_score:.3f} failed guardrail ({target_min_quality:.2f}) due to missing edges (Recall: {recall:.2f}). "
                f"Lowering high threshold from {last_cfg.high_threshold} -> {next_high} and low threshold from {last_cfg.low_threshold} -> {next_low}."
            )
            expected_outcome = "Increase edge recall by permitting weaker gradient boundaries to connect."

        # =========================================================================
        # RULE 3: Noise Blowout / Poor Precision (Too many false positive speckles)
        # =========================================================================
        elif noise_ratio > 0.15 or (has_gt and precision < 0.60):
            diagnosis = f"Noise contamination detected: {noise_ratio*100:.1f}% isolated pixels or low precision ({precision:.2f})."
            # Raise low threshold to filter noise, increase smoothing
            next_low = min(180, int(next_low * 1.5) + 15)
            if next_high <= next_low:
                next_high = next_low + 40
            if next_blur < 5:
                next_blur = 5
                next_sigma = 1.4
            elif next_blur == 5:
                next_sigma = min(2.5, next_sigma + 0.4)
            action_type = "SUPPRESS_NOISE"
            rationale = (
                f"Detected noisy edge map (Noise ratio: {noise_ratio*100:.1f}%, Precision: {precision:.2f}). "
                f"Raising low threshold from {last_cfg.low_threshold} -> {next_low} and increasing blur to kernel={next_blur}, sigma={next_sigma:.1f} to reject high-frequency noise."
            )
            expected_outcome = "Eliminate false-positive background noise pixels and clean up edge map."

        # =========================================================================
        # RULE 4: Candidate Accepted (Quality is solid) -> Optimize for Latency
        # =========================================================================
        elif last_verdict.is_acceptable:
            diagnosis = f"Quality guardrail satisfied ({primary_score:.3f} >= {target_min_quality:.2f}). Squeezing latency."
            # If current config has blur kernel > 1, try reducing blur kernel to eliminate Gaussian overhead
            if next_blur > 1:
                next_blur = 1  # test disabling blur for maximum CPU speedup
                next_sigma = 0.0
                action_type = "SPEEDUP_REDUCE_BLUR"
                rationale = (
                    f"Candidate passed quality criteria with score {primary_score:.3f}. "
                    f"Attempting to bypass Gaussian blur pre-filtering (kernel 1) to eliminate CPU convolution overhead."
                )
                expected_outcome = "Achieve significant speedup while testing whether edge quality remains above guardrail."
            elif next_l2:
                # Test switching from L2 norm to L1 norm (faster Euclidean approximation)
                next_l2 = False
                action_type = "SPEEDUP_L1_GRADIENT"
                rationale = (
                    f"Candidate passed quality criteria. Disabling L2gradient (switching to L1 norm approximation) "
                    f"to reduce floating-point magnitude calculations on CPU."
                )
                expected_outcome = "Reduce CPU cycle count for gradient calculation with minimal impact on edge geometry."
            else:
                # Fine-tune thresholds slightly around incumbent to find optimal trade-off
                next_high = min(300, int(next_high * 1.15))
                next_low = min(next_high - 10, int(next_low * 1.10))
                action_type = "FINE_TUNE_THRESHOLDS"
                rationale = (
                    f"Fine-tuning hysteresis thresholds ({next_low}, {next_high}) to thin edges and reduce downstream processing time."
                )
                expected_outcome = "Refine boundary sharpness while maintaining acceptable quality."

        # =========================================================================
        # RULE 5: Latency Regression (Slower without sufficient quality benefit)
        # =========================================================================
        else:
            diagnosis = f"Candidate was rejected (Verdict: {last_verdict.status}). Backtracking to explore alternate branch."
            if best_verdict is not None:
                base_ref = best_verdict.config
            else:
                base_ref = last_cfg

            # Shift kernel and adjust thresholds away from failed point
            next_blur = 3 if base_ref.blur_kernel != 3 else 5
            next_sigma = 1.0
            next_low = max(25, int(base_ref.low_threshold * 0.85))
            next_high = max(next_low + 25, int(base_ref.high_threshold * 0.90))
            action_type = "EXPLORE_ALTERNATE_BRANCH"
            rationale = (
                f"Previous configuration failed guardrail or suffered latency penalty. "
                f"Backtracking and branching to kernel={next_blur}, thresholds=({next_low}, {next_high})."
            )
            expected_outcome = "Escape dead-end parameter branch and seek acceptable trade-off space."

        # Safety & deduplication check against already tested configurations
        proposed = cls._sanitize_and_deduplicate(
            low=next_low,
            high=next_high,
            aperture=next_aperture,
            l2=next_l2,
            blur=next_blur,
            sigma=next_sigma,
            tested_configs=tested_configs,
            iteration=iteration
        )

        return AdaptiveDecision(
            proposed_config=proposed,
            diagnosis=diagnosis,
            rationale=rationale,
            expected_outcome=expected_outcome,
            action_type=action_type
        )

    @classmethod
    def _sanitize_and_deduplicate(
        cls,
        low: int,
        high: int,
        aperture: int,
        l2: bool,
        blur: int,
        sigma: float,
        tested_configs: List[EdgeDetectorConfig],
        iteration: int
    ) -> EdgeDetectorConfig:
        """Sanitizes bounds and alters parameters if an identical config was already tested."""
        low = int(max(10, min(240, low)))
        high = int(max(low + 15, min(350, high)))
        if blur % 2 == 0:
            blur = max(1, blur - 1)
        blur = max(1, min(15, blur))
        sigma = float(max(0.0, min(5.0, round(sigma, 2))))
        aperture = 3 if aperture not in (3, 5, 7) else aperture

        candidate = EdgeDetectorConfig(
            low_threshold=low,
            high_threshold=high,
            aperture_size=aperture,
            l2_gradient=l2,
            blur_kernel=blur,
            blur_sigma=sigma
        )

        # Check if already tested; if so, perturb slightly to maintain exploration
        already_tested = any(
            c.low_threshold == candidate.low_threshold and
            c.high_threshold == candidate.high_threshold and
            c.blur_kernel == candidate.blur_kernel and
            c.l2_gradient == candidate.l2_gradient
            for c in tested_configs
        )

        if already_tested:
            # Perturb thresholds to avoid redundant execution
            offset = 12 * (1 if iteration % 2 == 0 else -1)
            new_low = max(15, min(220, candidate.low_threshold + offset))
            new_high = max(new_low + 20, min(340, candidate.high_threshold + offset))
            candidate = EdgeDetectorConfig(
                low_threshold=new_low,
                high_threshold=new_high,
                aperture_size=aperture,
                l2_gradient=not l2 if iteration > 4 else l2,
                blur_kernel=blur,
                blur_sigma=sigma
            )

        return candidate
