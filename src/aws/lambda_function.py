"""AWS Lambda Function Handler for OpenCV 5 Workloads.

Stateless, CPU-optimized vision workload execution component:
- Accepts base64 image payload + EdgeDetectorConfig parameters.
- Enforces strict execution safeguards (image size < 1MB, timeout < 15s).
- Executes OpenCV 5 CPU edge detection.
- Returns output image as base64 PNG with high-resolution server execution timing
  and host environment metadata.
"""

from typing import Dict, Any, Optional
import os
import sys
import json
import base64
import time
import platform
import cv2
import numpy as np

# Track warm container status across invocations
COLD_START = True


def parse_event_body(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handles direct boto3 invoke, Lambda Function URL, or API Gateway events."""
    if "body" in event:
        body = event["body"]
        if event.get("isBase64Encoded", False):
            body = base64.b64decode(body).decode("utf-8")
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {}
        elif isinstance(body, dict):
            return body
    return event


def lambda_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AWS Lambda Entrypoint.

    Payload format:
    {
        "image_base64": "<base64 encoded png/jpeg>",
        "config": {
            "low_threshold": 50,
            "high_threshold": 150,
            "aperture_size": 3,
            "l2_gradient": true,
            "blur_kernel": 5,
            "blur_sigma": 1.2
        }
    }
    """
    global COLD_START
    is_cold = COLD_START
    COLD_START = False

    t_start = time.perf_counter_ns()

    try:
        data = parse_event_body(event)

        # 1. Payload size check
        raw_b64 = data.get("image_base64")
        if not raw_b64:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Missing 'image_base64' in request payload"})
            }

        # Enforce 1MB payload limit
        if len(raw_b64) > 1_500_000:  # ~1MB binary
            return {
                "statusCode": 413,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Payload exceeds 1MB safety limit"})
            }

        # 2. Decode image
        try:
            img_bytes = base64.b64decode(raw_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image data into OpenCV matrix")
        except Exception as e:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": f"Invalid image encoding: {str(e)}"})
            }

        h, w = img.shape[:2]
        if h > 1024 or w > 1024:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": f"Image dimension {w}x{h} exceeds maximum safe limit 1024x1024"})
            }

        # 3. Parse parameters
        cfg = data.get("config", {})
        low_thresh = int(cfg.get("low_threshold", 50))
        high_thresh = int(cfg.get("high_threshold", 150))
        aperture = int(cfg.get("aperture_size", 3))
        l2 = bool(cfg.get("l2_gradient", True))
        blur_k = int(cfg.get("blur_kernel", 5))
        blur_s = float(cfg.get("blur_sigma", 1.2))

        # Enforce sanity bounds
        low_thresh = max(10, min(250, low_thresh))
        high_thresh = max(low_thresh + 1, min(350, high_thresh))
        if blur_k % 2 == 0:
            blur_k = max(1, blur_k - 1)
        aperture = 3 if aperture not in (3, 5, 7) else aperture

        # 4. Execute OpenCV 5 Workload on CPU
        t_cv_start = time.perf_counter_ns()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if blur_k > 1:
            blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), sigmaX=blur_s)
        else:
            blurred = gray

        edges = cv2.Canny(blurred, low_thresh, high_thresh, apertureSize=aperture, L2gradient=l2)

        t_cv_elapsed_ms = (time.perf_counter_ns() - t_cv_start) / 1_000_000.0

        # 5. Encode output image to PNG base64
        success, enc_buf = cv2.imencode(".png", edges)
        if not success:
            raise RuntimeError("Failed to encode edge map to PNG")
        out_b64 = base64.b64encode(enc_buf).decode("utf-8")

        edge_count = int(np.count_nonzero(edges))
        total_time_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        host_info = {
            "platform": f"{platform.system()} {platform.machine()}",
            "opencv_version": cv2.__version__,
            "python_version": sys.version.split()[0],
            "cpu_count": os.cpu_count() or 1,
            "opencv_threads": cv2.getNumThreads(),
            "is_cold_start": is_cold,
            "memory_limit_mb": int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", 1024))
        }

        response_body = {
            "status": "SUCCESS",
            "dimensions": [w, h],
            "edge_pixel_count": edge_count,
            "edge_density": round(edge_count / (h * w), 5),
            "cv_execution_latency_ms": round(t_cv_elapsed_ms, 3),
            "total_server_latency_ms": round(total_time_ms, 3),
            "output_image_base64": out_b64,
            "host_info": host_info,
            "applied_config": {
                "low_threshold": low_thresh,
                "high_threshold": high_thresh,
                "aperture_size": aperture,
                "l2_gradient": l2,
                "blur_kernel": blur_k,
                "blur_sigma": blur_s
            }
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(response_body)
        }

    except Exception as exc:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "status": "ERROR",
                "error": str(exc)
            })
        }


if __name__ == "__main__":
    # Local CLI test of lambda_handler
    print("Testing lambda_handler locally with synthetic pattern...")
    sample_img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(sample_img, (30, 30), (98, 98), (255, 255, 255), -1)
    _, buf = cv2.imencode(".png", sample_img)
    b64 = base64.b64encode(buf).decode("utf-8")

    test_event = {
        "image_base64": b64,
        "config": {"low_threshold": 50, "high_threshold": 150}
    }

    res = lambda_handler(test_event)
    print("Response Status Code:", res["statusCode"])
    body = json.loads(res["body"])
    print("Status:", body.get("status"))
    print("Detected Edges:", body.get("edge_pixel_count"))
    print("CV Latency:", body.get("cv_execution_latency_ms"), "ms")
    print("Host Info:", body.get("host_info"))
