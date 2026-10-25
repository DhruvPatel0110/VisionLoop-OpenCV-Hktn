"""AWS Cloud Workload Executor & Client Bridge.

Provides a unified interface to execute the OpenCV 5 workload remotely on AWS or locally:
  - AWS_HTTP: Direct invocation via AWS Lambda Function URL or API Gateway
  - AWS_BOTO3: Invocation via AWS SDK (boto3 lambda client)
  - LOCAL_CONTAINER: Invocation via local Docker container (Lambda RIE at http://localhost:8080)
  - LOCAL_SIMULATION: In-process fallback execution using lambda_handler
"""

from typing import Dict, Any, Optional, Tuple
import os
import sys
import json
import base64
import time
import requests
import cv2
import numpy as np
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.vision.edge_detector import EdgeDetectorConfig
from src.aws.lambda_function import lambda_handler


class CloudExecutionResult(BaseModel):
    """Result of an AWS cloud execution."""
    status: str
    edge_pixel_count: int
    edge_density: float
    cv_execution_latency_ms: float
    roundtrip_latency_ms: float
    dimensions: Tuple[int, int]
    mode_used: str
    host_info: Dict[str, Any]
    applied_config: Dict[str, Any]
    error_message: Optional[str] = None


class CloudExecutor:
    """Dispatches OpenCV 5 workloads to AWS Lambda or local execution target."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        function_name: Optional[str] = None,
        timeout_seconds: float = 15.0
    ):
        # Priority: explicit arg -> environment variable
        self.endpoint_url = endpoint_url or os.environ.get("AWS_LAMBDA_ENDPOINT_URL")
        self.function_name = function_name or os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "visionloop-cv5")
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        image: np.ndarray,
        config: Optional[EdgeDetectorConfig] = None
    ) -> Tuple[np.ndarray, CloudExecutionResult]:
        """
        Executes edge detection remotely on AWS (or fallback to local container / simulation).
        """
        if config is None:
            config = EdgeDetectorConfig()

        # Encode image to base64 PNG
        success, buf = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Failed to encode input image to PNG")
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # Check payload size (< 1MB)
        if len(img_b64) > 1_400_000:
            raise ValueError("Input image exceeds 1MB network payload limit for AWS execution")

        payload = {
            "image_base64": img_b64,
            "config": config.model_dump()
        }

        t0 = time.perf_counter()

        # 1. Try remote AWS Lambda Function URL / API Gateway if configured
        if self.endpoint_url:
            return self._invoke_http(self.endpoint_url, payload, t0, "AWS_HTTP")

        # 2. Try AWS boto3 invoke if credentials are present
        if self._has_aws_credentials():
            try:
                return self._invoke_boto3(payload, t0)
            except Exception as e:
                print(f"[Warning] Boto3 invocation failed ({e}), falling back to local simulation.")

        # 3. Fallback: In-process Lambda simulation (exact Lambda handler execution)
        return self._invoke_simulation(payload, t0)

    def _invoke_http(
        self,
        url: str,
        payload: Dict[str, Any],
        t0: float,
        mode_label: str
    ) -> Tuple[np.ndarray, CloudExecutionResult]:
        """Calls Lambda via HTTP POST with automatic retry."""
        headers = {"Content-Type": "application/json"}
        last_err = None

        for attempt in range(1, 4):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
                roundtrip_ms = (time.perf_counter() - t0) * 1000.0

                if resp.status_code == 200:
                    data = resp.json()
                    # Handle API gateway nested body
                    if isinstance(data.get("body"), str):
                        data = json.loads(data["body"])
                    return self._parse_success_response(data, roundtrip_ms, mode_label)
                else:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as e:
                last_err = e
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"Failed to execute on AWS after 3 attempts: {last_err}")

    def _invoke_boto3(
        self,
        payload: Dict[str, Any],
        t0: float
    ) -> Tuple[np.ndarray, CloudExecutionResult]:
        """Invokes Lambda via AWS boto3 client."""
        import boto3
        client = boto3.client("lambda")
        resp = client.invoke(
            FunctionName=self.function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        resp_payload = json.loads(resp["Payload"].read().decode("utf-8"))

        if "body" in resp_payload and isinstance(resp_payload["body"], str):
            resp_payload = json.loads(resp_payload["body"])

        return self._parse_success_response(resp_payload, roundtrip_ms, "AWS_BOTO3")

    def _invoke_simulation(
        self,
        payload: Dict[str, Any],
        t0: float
    ) -> Tuple[np.ndarray, CloudExecutionResult]:
        """Invokes Lambda handler locally in-process."""
        res = lambda_handler(payload)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0

        if res["statusCode"] != 200:
            err_data = json.loads(res.get("body", "{}"))
            raise RuntimeError(f"Simulation failed ({res['statusCode']}): {err_data.get('error')}")

        data = json.loads(res["body"])
        return self._parse_success_response(data, roundtrip_ms, "LOCAL_SIMULATION")

    def _parse_success_response(
        self,
        data: Dict[str, Any],
        roundtrip_ms: float,
        mode_label: str
    ) -> Tuple[np.ndarray, CloudExecutionResult]:
        """Decodes base64 edge map and builds CloudExecutionResult."""
        out_b64 = data["output_image_base64"]
        img_bytes = base64.b64decode(out_b64)
        edges = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        dims = tuple(data.get("dimensions", (edges.shape[1], edges.shape[0])))

        result = CloudExecutionResult(
            status="SUCCESS",
            edge_pixel_count=data.get("edge_pixel_count", int(np.count_nonzero(edges))),
            edge_density=data.get("edge_density", 0.0),
            cv_execution_latency_ms=data.get("cv_execution_latency_ms", 0.0),
            roundtrip_latency_ms=round(roundtrip_ms, 3),
            dimensions=dims,
            mode_used=mode_label,
            host_info=data.get("host_info", {}),
            applied_config=data.get("applied_config", {})
        )

        return edges, result

    @staticmethod
    def _has_aws_credentials() -> bool:
        """Checks if AWS credentials exist in environment or profile."""
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            return True
        try:
            import boto3
            return bool(boto3.Session().get_credentials())
        except Exception:
            return False
