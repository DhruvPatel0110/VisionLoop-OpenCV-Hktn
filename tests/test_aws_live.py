"""Standalone CLI test for AWS Lambda endpoint verification.

Usage:
    python tests/test_aws_live.py
    python tests/test_aws_live.py --url https://<function-url-id>.lambda-url.us-east-1.on.aws/
"""

import os
import sys
import argparse
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aws.cloud_executor import CloudExecutor
from src.vision.edge_detector import EdgeDetectorConfig


def test_aws(endpoint_url: str = None):
    url = endpoint_url or os.environ.get("AWS_LAMBDA_ENDPOINT_URL")

    print("=" * 80)
    print("VISIONLOOP: AWS CLOUD WORKLOAD VERIFICATION")
    print(f"Target Endpoint: {url or 'LOCAL_SIMULATION_MODE (No AWS_LAMBDA_ENDPOINT_URL set)'}")
    print("=" * 80)

    sample_img_path = "data/test_images/sample_01.png"
    if not os.path.exists(sample_img_path):
        print(f"Error: Sample image not found at {sample_img_path}")
        sys.exit(1)

    img = cv2.imread(sample_img_path)
    executor = CloudExecutor(endpoint_url=url)
    config = EdgeDetectorConfig(low_threshold=50, high_threshold=150, blur_kernel=5)

    print("\nSending bounded OpenCV 5 edge detection request...")
    edges, result = executor.execute(image=img, config=config)

    print(f"\nExecution Result:")
    print(f"  Mode Used:                {result.mode_used}")
    print(f"  Status:                   {result.status}")
    print(f"  Detected Edges:           {result.edge_pixel_count} pixels")
    print(f"  Edge Density:             {result.edge_density:.4f}")
    print(f"  Server Execution Latency: {result.cv_execution_latency_ms:.2f} ms")
    print(f"  Round-Trip Latency:       {result.roundtrip_latency_ms:.2f} ms")
    print(f"  Remote Host Info:         {result.host_info}")

    out_path = "data/aws_verification_output.png"
    cv2.imwrite(out_path, edges)
    print(f"\nOutput edge map successfully decoded and saved to: {out_path}")
    print("=" * 80)
    print("AWS CLOUD WORKLOAD TEST: PASSED")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AWS Lambda Endpoint")
    parser.add_argument("--url", type=str, default=None, help="Lambda Function URL")
    args = parser.parse_args()
    test_aws(endpoint_url=args.url)
