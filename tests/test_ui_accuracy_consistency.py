"""Automated Validation Tests for VisionLoop UI Accuracy, Consistency & Usability.

Validates:
1. Best-candidate selection (incumbent identification, is_final_winner uniqueness).
2. Dynamic metric labeling (Ground Truth F1 vs Proxy Quality).
3. Realistic stopping reason formulation in status_summary.
4. Latency and speedup formulas.
5. Handling of runs where quality threshold is unmet.
"""

import os
import sys
import base64
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.web.server import app

client = TestClient(app)


class TestUIAccuracyAndConsistency:
    """Test suite ensuring frontend-backend data contracts meet accuracy standards."""

    def test_ground_truth_mode_evaluation(self):
        """When dataset has ground truth, metrics must be GROUND_TRUTH_F1 with valid reference."""
        payload = {
            "preset_id": "sample_04_low_contrast",
            "execution_mode": "local",
            "max_iterations": 3,
            "target_min_quality": 0.60
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Check Ground Truth mode attributes
        assert data["has_ground_truth"] is True
        assert data["baseline_quality_type"] == "GROUND_TRUTH_F1"
        assert data["preview_reference_b64"] is not None
        assert data["preview_reference_b64"].startswith("data:image/png;base64,")

        # Winner checks
        assert "status_summary" in data
        assert "Iteration budget reached" in data["status_summary"]

        winner_traces = [t for t in data["traces"] if t.get("is_final_winner")]
        if data["best_config"] is not None:
            assert len(winner_traces) == 1, "There should be exactly one final winner trace"
            assert data["winner_iteration"] is not None
            assert data["speedup_factor"] is not None
            assert "retained" in data["status_summary"]
        else:
            assert len(winner_traces) == 0
            assert "No candidate satisfied" in data["status_summary"]

    def test_unlabelled_proxy_mode_evaluation(self):
        """When user uploads custom image without GT, metrics must be PROXY_COMPOSITE and reference must be None."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, (255, 255, 255), 2)
        _, buf = cv2.imencode(".png", img)
        b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

        payload = {
            "image_base64": b64,
            "execution_mode": "local",
            "max_iterations": 3,
            "target_min_quality": 0.60
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Check Proxy Quality mode attributes
        assert data["has_ground_truth"] is False
        assert data["baseline_quality_type"] == "PROXY_COMPOSITE"
        assert data["preview_reference_b64"] is None, "Proxy mode must NOT show fabricated reference"

        # Check that metrics are within valid probability / heuristic range [0, 1]
        assert 0.0 <= data["baseline_quality_score"] <= 1.0
        for t in data["traces"]:
            assert 0.0 <= t["verdict"]["primary_quality_score"] <= 1.0

    def test_unmet_quality_guardrail_handles_no_winner(self):
        """When quality target cannot be met (0.95 on noisy image), best candidate must be None and stopping reason explicit."""
        payload = {
            "preset_id": "sample_03_noisy",
            "execution_mode": "local",
            "max_iterations": 2,
            "target_min_quality": 0.95  # Within valid range [0.30, 0.95], but unmet on sample_03_noisy
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        # Best candidate must not be falsely fabricated
        assert data["best_config"] is None
        assert data["winner_iteration"] is None
        assert data["preview_best_b64"] is None
        assert "No candidate satisfied quality guardrail" in data["status_summary"]

        # No trace should be flagged as final winner
        winner_traces = [t for t in data["traces"] if t.get("is_final_winner")]
        assert len(winner_traces) == 0

    def test_speedup_and_latency_formulas(self):
        """Verify speedup and latency reduction mathematical relationships."""
        payload = {
            "preset_id": "sample_04_low_contrast",
            "execution_mode": "local",
            "max_iterations": 3,
            "target_min_quality": 0.50
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        base_lat = data["baseline_latency_ms"]
        assert base_lat > 0

        if data["best_latency_ms"] is not None:
            best_lat = data["best_latency_ms"]
            expected_speedup = base_lat / best_lat
            expected_reduction = ((base_lat - best_lat) / base_lat) * 100.0

            assert abs(data["speedup_factor"] - expected_speedup) < 1e-3
            assert abs(data["latency_reduction_pct"] - expected_reduction) < 0.05
