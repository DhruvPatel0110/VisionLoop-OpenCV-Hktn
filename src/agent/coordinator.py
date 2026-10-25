"""VisionLoop Agent Coordinator - Autonomous Optimization Loop & Audit Trace Manager.

Manages:
1. Agent state machine (BASELINE -> PROPOSE -> EXECUTE -> EVALUATE -> DECIDE -> NEXT/STOP).
2. Experiment budget enforcement (max iterations & wall-clock timeout).
3. Structured audit trail logging and evidence export to JSON and Markdown.
"""

from typing import List, Optional, Dict, Any, Tuple
import os
import sys
import time
import json
import uuid
from enum import Enum
import cv2
import numpy as np
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.edge_detector import EdgeDetector, EdgeDetectorConfig
from src.eval.evaluator import GroundedEvaluator, EvaluationVerdict, QualityMetrics, TimingMetrics
from src.agent.optimizer import AdaptiveOptimizer, AdaptiveDecision


class AgentState(str, Enum):
    IDLE = "IDLE"
    RUNNING_BASELINE = "RUNNING_BASELINE"
    PROPOSING = "PROPOSING"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    DECIDING = "DECIDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IterationTrace(BaseModel):
    """Step-by-step trace record of a single autonomous experiment."""
    iteration: int
    timestamp_utc: str
    action_type: str
    diagnosis: str
    decision_rationale: str
    expected_outcome: str
    config: EdgeDetectorConfig
    verdict: EvaluationVerdict
    is_incumbent_best: bool = False
    is_final_winner: bool = False
    output_image_path: Optional[str] = None


class OptimizationSummary(BaseModel):
    """Final summary of the autonomous optimization loop."""
    session_id: str
    image_id: str
    timestamp_utc: str
    state: AgentState
    stop_reason: str
    status_summary: str = ""
    winner_iteration: Optional[int] = None
    budget_max_iterations: int
    iterations_completed: int
    total_elapsed_seconds: float
    target_min_quality: float

    # Baseline & Best results
    has_ground_truth: bool = False
    baseline_config: EdgeDetectorConfig
    baseline_latency_ms: float
    baseline_quality_score: float
    baseline_quality_type: str

    best_config: Optional[EdgeDetectorConfig] = None
    best_latency_ms: Optional[float] = None
    best_quality_score: Optional[float] = None
    speedup_factor: Optional[float] = None
    latency_reduction_pct: Optional[float] = None
    quality_delta: Optional[float] = None

    candidates_accepted: int = 0
    candidates_rejected_quality: int = 0
    candidates_rejected_latency: int = 0

    trace_json_path: Optional[str] = None
    trace_markdown_path: Optional[str] = None
    traces: List[IterationTrace] = Field(default_factory=list)


class VisionLoopCoordinator:
    """Orchestrates autonomous testing and parameter tuning for OpenCV 5 workloads."""

    def __init__(
        self,
        max_iterations: int = 6,
        max_seconds: float = 30.0,
        target_min_quality: float = 0.70,
        output_dir: str = "runs"
    ):
        self.max_iterations = max_iterations
        self.max_seconds = max_seconds
        self.target_min_quality = target_min_quality
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        image: np.ndarray,
        reference_edges: Optional[np.ndarray] = None,
        image_id: str = "test_image",
        save_candidate_images: bool = True
    ) -> OptimizationSummary:
        """
        Executes the autonomous loop end-to-end on a given image.
        """
        session_id = f"opt_{uuid.uuid4().hex[:8]}"
        session_dir = os.path.join(self.output_dir, session_id).replace("\\", "/")
        os.makedirs(session_dir, exist_ok=True)

        t_session_start = time.perf_counter()
        traces: List[IterationTrace] = []
        tested_configs: List[EdgeDetectorConfig] = []

        print("=" * 80)
        print(f"VISIONLOOP AUTONOMOUS AGENT STARTED [Session: {session_id}]")
        print(f"Target Image: {image_id} | Max Iterations: {self.max_iterations} | Timeout: {self.max_seconds}s")
        print(f"Quality Guardrail Threshold: {self.target_min_quality}")
        print("=" * 80)

        # Save input and reference images for complete auditability
        cv2.imwrite(os.path.join(session_dir, "input_original.png"), image)
        if reference_edges is not None:
            cv2.imwrite(os.path.join(session_dir, "reference_ground_truth.png"), reference_edges)

        # -------------------------------------------------------------
        # STEP 1: Execute Baseline
        # -------------------------------------------------------------
        baseline_cfg = EdgeDetector.get_baseline_config()
        tested_configs.append(baseline_cfg)

        print("\n[Step 1] Running Baseline Workload...")
        baseline_edges, baseline_verdict = GroundedEvaluator.evaluate_candidate(
            image=image,
            candidate_config=baseline_cfg,
            reference_edges=reference_edges,
            min_quality_threshold=self.target_min_quality
        )

        base_lat = baseline_verdict.timing_metrics.median_latency_ms
        base_score = baseline_verdict.primary_quality_score
        base_metric_type = baseline_verdict.quality_metric_type

        # Save baseline image
        base_img_path = os.path.join(session_dir, "iter_00_baseline.png").replace("\\", "/")
        cv2.imwrite(base_img_path, baseline_edges)

        print(f"         Baseline Median Latency: {base_lat:.2f} ms")
        print(f"         Baseline Quality Score:  {base_score:.4f} ({base_metric_type})")
        print(f"         Baseline Pass Status:    {baseline_verdict.status}")

        # Incumbent tracker
        incumbent_config: Optional[EdgeDetectorConfig] = baseline_cfg if baseline_verdict.is_acceptable else None
        incumbent_verdict: Optional[EvaluationVerdict] = baseline_verdict if baseline_verdict.is_acceptable else None
        last_verdict: EvaluationVerdict = baseline_verdict

        accepted_count = 1 if baseline_verdict.is_acceptable else 0
        rejected_quality_count = 1 if baseline_verdict.status == "REJECTED_QUALITY" else 0
        rejected_latency_count = 0

        # Record Iteration 0 (Baseline)
        traces.append(IterationTrace(
            iteration=0,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            action_type="BASELINE_EXECUTION",
            diagnosis="Established benchmark baseline metrics.",
            decision_rationale="Initial baseline configuration run to measure benchmark latency and quality reference.",
            expected_outcome="Establish ground-truth latency and edge score for comparison.",
            config=baseline_cfg,
            verdict=baseline_verdict,
            is_incumbent_best=baseline_verdict.is_acceptable,
            output_image_path=base_img_path
        ))

        # -------------------------------------------------------------
        # STEP 2: Autonomous Adaptive Feedback Loop
        # -------------------------------------------------------------
        stop_reason = "MAX_ITERATIONS_REACHED"

        for iteration in range(1, self.max_iterations + 1):
            elapsed = time.perf_counter() - t_session_start
            if elapsed >= self.max_seconds:
                stop_reason = f"TIME_BUDGET_EXHAUSTED ({elapsed:.1f}s >= {self.max_seconds}s)"
                print(f"\n[Warning] {stop_reason}. Terminating loop.")
                break

            print(f"\n[Iteration {iteration}/{self.max_iterations}] (Elapsed: {elapsed:.2f}s)")

            # PROPOSE: Adaptive decision engine evaluates previous result
            decision: AdaptiveDecision = AdaptiveOptimizer.propose_next_configuration(
                last_verdict=last_verdict,
                best_verdict=incumbent_verdict,
                tested_configs=tested_configs,
                iteration=iteration,
                target_min_quality=self.target_min_quality
            )
            candidate_cfg = decision.proposed_config
            tested_configs.append(candidate_cfg)

            print(f"  Diagnosis: {decision.diagnosis}")
            print(f"  Rationale: {decision.rationale}")
            print(f"  Testing Config: low={candidate_cfg.low_threshold}, high={candidate_cfg.high_threshold}, blur={candidate_cfg.blur_kernel}, l2={candidate_cfg.l2_gradient}")

            # EXECUTE & EVALUATE: Run workload with candidate parameters
            cand_edges, cand_verdict = GroundedEvaluator.evaluate_candidate(
                image=image,
                candidate_config=candidate_cfg,
                reference_edges=reference_edges,
                baseline_timing=baseline_verdict.timing_metrics,
                baseline_is_acceptable=baseline_verdict.is_acceptable,
                min_quality_threshold=self.target_min_quality
            )

            cand_lat = cand_verdict.timing_metrics.median_latency_ms
            cand_score = cand_verdict.primary_quality_score
            speedup = cand_verdict.speedup_factor or 1.0

            # Save candidate image
            cand_img_path = None
            if save_candidate_images:
                cand_img_path = os.path.join(session_dir, f"iter_{iteration:02d}_{cand_verdict.status.lower()}.png").replace("\\", "/")
                cv2.imwrite(cand_img_path, cand_edges)

            # Update counters
            if cand_verdict.status == "ACCEPTED":
                accepted_count += 1
            elif cand_verdict.status == "REJECTED_QUALITY":
                rejected_quality_count += 1
            else:
                rejected_latency_count += 1

            # DECIDE: Winner updating logic
            is_new_best = False
            if cand_verdict.is_acceptable:
                if incumbent_verdict is None:
                    # Baseline had failed quality, this is the first acceptable candidate!
                    is_new_best = True
                    incumbent_config = candidate_cfg
                    incumbent_verdict = cand_verdict
                    print(f"  >>> FIRST ACCEPTABLE CANDIDATE FOUND! (Score: {cand_score:.4f}, Latency: {cand_lat:.2f} ms)")
                elif cand_lat < incumbent_verdict.timing_metrics.median_latency_ms:
                    # Faster than previous best while still acceptable!
                    is_new_best = True
                    incumbent_config = candidate_cfg
                    incumbent_verdict = cand_verdict
                    print(f"  >>> NEW BEST CANDIDATE! Speedup: {speedup:.2f}x ({cand_lat:.2f} ms vs {incumbent_verdict.timing_metrics.median_latency_ms:.2f} ms)")
                elif incumbent_verdict.primary_quality_score < self.target_min_quality and cand_score > incumbent_verdict.primary_quality_score:
                    is_new_best = True
                    incumbent_config = candidate_cfg
                    incumbent_verdict = cand_verdict
                    print(f"  >>> QUALITY RECOVERY BEST! Score: {cand_score:.4f}")

            status_str = f"PASS ({cand_score:.4f}, {cand_lat:.2f}ms, {speedup:.2f}x)" if cand_verdict.is_acceptable else f"REJECTED: {', '.join(cand_verdict.rejection_reasons)}"
            print(f"  Result:    {status_str}")

            # Append trace
            traces.append(IterationTrace(
                iteration=iteration,
                timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                action_type=decision.action_type,
                diagnosis=decision.diagnosis,
                decision_rationale=decision.rationale,
                expected_outcome=decision.expected_outcome,
                config=candidate_cfg,
                verdict=cand_verdict,
                is_incumbent_best=is_new_best,
                output_image_path=cand_img_path
            ))

            last_verdict = cand_verdict

        # -------------------------------------------------------------
        # STEP 3: Compile Final Report & Summary
        # -------------------------------------------------------------
        total_time = round(time.perf_counter() - t_session_start, 3)

        best_lat = incumbent_verdict.timing_metrics.median_latency_ms if incumbent_verdict else None
        best_score = incumbent_verdict.primary_quality_score if incumbent_verdict else None
        final_speedup = round(base_lat / best_lat, 3) if best_lat and base_lat > 0 else None
        lat_red_pct = round(((base_lat - best_lat) / base_lat) * 100.0, 2) if best_lat and base_lat > 0 else None
        qual_delta = round(best_score - base_score, 4) if best_score is not None else None

        # Determine unique final winning candidate
        winner_iter = None
        if incumbent_config is not None:
            # Find the last trace whose config matches incumbent_config and is acceptable
            for t in reversed(traces):
                if t.config == incumbent_config and t.verdict.is_acceptable:
                    t.is_final_winner = True
                    winner_iter = t.iteration
                    break
        for t in traces:
            if t.iteration != winner_iter:
                t.is_final_winner = False

        # Build precise, unambiguous status summary
        if incumbent_config is not None:
            if stop_reason == "MAX_ITERATIONS_REACHED":
                if final_speedup and final_speedup > 1.0:
                    status_summary = f"Iteration budget reached. Best validated candidate retained (Iter #{winner_iter:02d}, {final_speedup:.2f}x speedup)."
                else:
                    status_summary = f"Iteration budget reached. Best validated candidate retained (Iter #{winner_iter:02d})."
            elif "TIME_BUDGET" in stop_reason:
                status_summary = f"Time budget reached. Best validated candidate retained (Iter #{winner_iter:02d})."
            else:
                status_summary = f"Optimization completed ({stop_reason}). Validated candidate found (Iter #{winner_iter:02d})."
        else:
            if stop_reason == "MAX_ITERATIONS_REACHED":
                status_summary = f"Iteration budget reached. No candidate satisfied quality guardrail threshold ({self.target_min_quality:.2f})."
            else:
                status_summary = f"Optimization finished ({stop_reason}). No acceptable candidate met quality guardrail."

        trace_json_path = os.path.join(session_dir, "experiment_trace.json").replace("\\", "/")
        trace_md_path = os.path.join(session_dir, "experiment_report.md").replace("\\", "/")

        summary = OptimizationSummary(
            session_id=session_id,
            image_id=image_id,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            state=AgentState.COMPLETED,
            stop_reason=stop_reason,
            status_summary=status_summary,
            winner_iteration=winner_iter,
            budget_max_iterations=self.max_iterations,
            iterations_completed=len(traces) - 1,
            total_elapsed_seconds=total_time,
            target_min_quality=self.target_min_quality,
            has_ground_truth=(reference_edges is not None),
            baseline_config=baseline_cfg,
            baseline_latency_ms=base_lat,
            baseline_quality_score=base_score,
            baseline_quality_type=base_metric_type,
            best_config=incumbent_config,
            best_latency_ms=best_lat,
            best_quality_score=best_score,
            speedup_factor=final_speedup,
            latency_reduction_pct=lat_red_pct,
            quality_delta=qual_delta,
            candidates_accepted=accepted_count,
            candidates_rejected_quality=rejected_quality_count,
            candidates_rejected_latency=rejected_latency_count,
            trace_json_path=trace_json_path,
            trace_markdown_path=trace_md_path,
            traces=traces
        )

        # Write JSON trace
        with open(trace_json_path, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(), f, indent=2)

        # Write Markdown report
        self._write_markdown_report(summary, trace_md_path)

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print(f"Total Iterations: {len(traces)-1} | Time: {total_time:.2f}s | Reason: {stop_reason}")
        if incumbent_config:
            print(f"Winner Config: low={incumbent_config.low_threshold}, high={incumbent_config.high_threshold}, blur={incumbent_config.blur_kernel}, l2={incumbent_config.l2_gradient}")
            print(f"Speedup Factor: {final_speedup}x ({base_lat:.2f} ms -> {best_lat:.2f} ms | {lat_red_pct}% latency reduction)")
            print(f"Quality Score:  {best_score:.4f} (Baseline: {base_score:.4f}, Delta: {qual_delta:+.4f})")
        else:
            print("Notice: No tested configuration improved on baseline within quality constraints.")
        print(f"Trace saved to: {trace_json_path}")
        print(f"Report saved to: {trace_md_path}")
        print("=" * 80)

        return summary

    def _write_markdown_report(self, summary: OptimizationSummary, path: str):
        """Generates a clean GitHub-flavored markdown report."""
        md = []
        md.append(f"# VisionLoop Experiment Report - Session `{summary.session_id}`\n")
        md.append(f"- **Image ID:** `{summary.image_id}`")
        md.append(f"- **Timestamp:** `{summary.timestamp_utc}`")
        md.append(f"- **Stop Condition:** `{summary.stop_reason}`")
        md.append(f"- **Status Summary:** {summary.status_summary}")
        md.append(f"- **Evaluation Mode:** `{summary.baseline_quality_type}` (Ground Truth: `{summary.has_ground_truth}`)")
        md.append(f"- **Total Time:** `{summary.total_elapsed_seconds:.2f} seconds`\n")

        md.append("## Executive Results Summary\n")
        md.append("| Metric | Baseline | Best Validated | Improvement |")
        md.append("| :--- | :--- | :--- | :--- |")
        b_lat = f"{summary.baseline_latency_ms:.2f} ms"
        w_lat = f"{summary.best_latency_ms:.2f} ms" if summary.best_latency_ms else "N/A"
        speedup = f"**{summary.speedup_factor:.2f}x** ({summary.latency_reduction_pct:+.1f}%)" if summary.speedup_factor else "N/A"
        md.append(f"| **Median CPU Latency** | {b_lat} | {w_lat} | {speedup} |")

        b_sc = f"{summary.baseline_quality_score:.4f}"
        w_sc = f"{summary.best_quality_score:.4f}" if summary.best_quality_score else "N/A"
        delta = f"{summary.quality_delta:+.4f}" if summary.quality_delta is not None else "N/A"
        qual_label = "Ground Truth Edge F1" if summary.has_ground_truth else "Composite Proxy Quality"
        md.append(f"| **{qual_label}** | {b_sc} | {w_sc} | {delta} |")

        md.append(f"\n- **Candidates Accepted:** `{summary.candidates_accepted}`")
        md.append(f"- **Rejected for Quality:** `{summary.candidates_rejected_quality}`")
        md.append(f"- **Rejected for Latency:** `{summary.candidates_rejected_latency}`\n")

        md.append("## Autonomous Perception-Action Decision Trace\n")
        md.append("| Iter | Action / Rationale | Config | Latency (ms) | Quality Score | Verdict |")
        md.append("| :---: | :--- | :--- | :---: | :---: | :---: |")

        for t in summary.traces:
            c = t.config
            cfg_str = f"low={c.low_threshold}, hi={c.high_threshold}, blur={c.blur_kernel}"
            lat_str = f"{t.verdict.timing_metrics.median_latency_ms:.2f}"
            sc_str = f"{t.verdict.primary_quality_score:.4f}"
            badge = f"**WINNER**" if t.is_final_winner else t.verdict.status
            md.append(f"| {t.iteration} | **{t.action_type}**<br>_{t.decision_rationale}_ | `{cfg_str}` | {lat_str} | {sc_str} | {badge} |")

        md.append("\n---\n_Generated autonomously by VisionLoop Agent._\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
