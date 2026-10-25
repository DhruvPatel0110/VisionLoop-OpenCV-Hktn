"""Unit and Integration Tests for Phase 4: Web API Backend and Dashboard endpoints."""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.web.server import app

client = TestClient(app)


class TestWebAPI:
    """Test suite for FastAPI endpoints."""

    def test_get_tasks_and_presets(self):
        resp = client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert len(data["supported_tasks"]) >= 1
        assert len(data["presets"]) >= 5
        assert any(p["id"] == "sample_04_low_contrast" for p in data["presets"])

    def test_get_preset_image(self):
        resp = client.get("/api/preset-image/sample_01/original")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100

    def test_run_optimization_preset(self):
        payload = {
            "preset_id": "sample_04_low_contrast",
            "execution_mode": "local",
            "max_iterations": 3,
            "target_min_quality": 0.70
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["state"] == "COMPLETED"
        assert len(data["traces"]) >= 3
        assert "preview_original_b64" in data
        assert "preview_baseline_b64" in data
        assert "preview_best_b64" in data
        assert data["preview_original_b64"].startswith("data:image/png;base64,")

        session_id = data["session_id"]

        # Test trace endpoint
        trace_resp = client.get(f"/api/trace/{session_id}")
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["session_id"] == session_id

        # Test report download endpoint (markdown)
        report_resp = client.get(f"/api/report/{session_id}?format=markdown")
        assert report_resp.status_code == 200
        assert "VisionLoop Experiment Report" in report_resp.text

        # Test report download endpoint (json)
        json_resp = client.get(f"/api/report/{session_id}?format=json")
        assert json_resp.status_code == 200

    def test_static_html_served(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<title>VisionLoop" in resp.text
        assert "OpenCV 5" in resp.text

    def test_system_status(self):
        resp = client.get("/api/system/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "HEALTHY"
        assert "opencv_version" in data
        assert "aws_status" in data
        assert data["agent_status"] == "READY"

    def test_experiment_history(self):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert "history" in data

    def test_run_oversized_upload_auto_scales(self):
        import cv2
        import numpy as np
        import base64

        # Create 1536x1024 synthetic image matching the user's uploaded dimensions
        img = np.zeros((1024, 1536, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 200), (1000, 800), (255, 255, 255), -1)
        _, buf = cv2.imencode(".png", img)
        b64 = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

        payload = {
            "image_base64": b64,
            "execution_mode": "local",
            "max_iterations": 2,
            "target_min_quality": 0.50
        }
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "COMPLETED"
        assert len(data["traces"]) >= 2


