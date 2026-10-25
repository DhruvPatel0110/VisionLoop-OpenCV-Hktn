"""Unit and Integration Tests for Phase 3: AWS Lambda Handler and CloudExecutor."""

import os
import sys
import json
import base64
import pytest
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aws.lambda_function import lambda_handler
from src.aws.cloud_executor import CloudExecutor, CloudExecutionResult
from src.vision.edge_detector import EdgeDetectorConfig


class TestLambdaHandler:
    """Test the AWS Lambda entrypoint logic."""

    @pytest.fixture
    def sample_base64_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode("utf-8")

    def test_lambda_handler_success(self, sample_base64_image):
        event = {
            "image_base64": sample_base64_image,
            "config": {
                "low_threshold": 50,
                "high_threshold": 150,
                "aperture_size": 3,
                "l2_gradient": True,
                "blur_kernel": 5,
                "blur_sigma": 1.2
            }
        }
        res = lambda_handler(event)
        assert res["statusCode"] == 200
        body = json.loads(res["body"])
        assert body["status"] == "SUCCESS"
        assert body["dimensions"] == [100, 100]
        assert body["edge_pixel_count"] > 0
        assert "host_info" in body
        assert body["host_info"]["opencv_version"].startswith("5.")
        assert "output_image_base64" in body

        # Verify decoded image
        raw_out = base64.b64decode(body["output_image_base64"])
        out_img = cv2.imdecode(np.frombuffer(raw_out, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        assert out_img is not None
        assert out_img.shape == (100, 100)

    def test_lambda_handler_missing_image(self):
        event = {"config": {"low_threshold": 50}}
        res = lambda_handler(event)
        assert res["statusCode"] == 400
        body = json.loads(res["body"])
        assert "error" in body

    def test_lambda_handler_corrupt_image(self):
        event = {
            "image_base64": "not-valid-base64-data===",
            "config": {}
        }
        res = lambda_handler(event)
        assert res["statusCode"] == 400

    def test_lambda_handler_payload_too_large(self):
        # Create oversized payload string (> 1.5MB)
        oversized = "A" * 1_600_000
        event = {"image_base64": oversized}
        res = lambda_handler(event)
        assert res["statusCode"] == 413
        body = json.loads(res["body"])
        assert "1MB safety limit" in body["error"]


class TestCloudExecutor:
    """Test CloudExecutor client bridge."""

    def test_cloud_executor_simulation_mode(self):
        img_path = "data/test_images/sample_01.png"
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)

        executor = CloudExecutor()
        cfg = EdgeDetectorConfig(low_threshold=60, high_threshold=140)

        edges, result = executor.execute(image=img, config=cfg)

        assert isinstance(result, CloudExecutionResult)
        assert result.status == "SUCCESS"
        assert result.mode_used == "LOCAL_SIMULATION"
        assert edges.shape == (512, 512)
        assert result.edge_pixel_count > 0
        assert result.roundtrip_latency_ms > 0
        assert result.host_info["opencv_version"].startswith("5.")

    def test_cloud_executor_rejects_oversized_before_network(self):
        # Create huge dummy array that would exceed payload cap
        oversized_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Add random noise so PNG compression can't compress it below 1MB
        np.random.seed(0)
        oversized_img[:] = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        executor = CloudExecutor()
        with pytest.raises(ValueError, match="1MB network payload limit"):
            executor.execute(oversized_img)


class TestAWSLiveEndpoint:
    """Test live remote AWS endpoint if environment variable is configured."""

    def test_live_aws_invocation_if_configured(self):
        endpoint_url = os.environ.get("AWS_LAMBDA_ENDPOINT_URL")
        if not endpoint_url:
            pytest.skip("AWS_LAMBDA_ENDPOINT_URL not set. Skipping remote live AWS call.")

        img_path = "data/test_images/sample_01.png"
        img = cv2.imread(img_path)
        executor = CloudExecutor(endpoint_url=endpoint_url)

        edges, result = executor.execute(image=img)
        assert result.status == "SUCCESS"
        assert result.mode_used == "AWS_HTTP"
        assert edges.shape == (512, 512)
        print(f"Live AWS execution returned in {result.roundtrip_latency_ms:.2f} ms")
