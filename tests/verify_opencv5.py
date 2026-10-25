"""OpenCV 5 Environment Verification and API Health Check Script.

This script acts as the Phase 0 stop-the-line gatekeeper:
1. Asserts cv2.__version__ starts with '5.'
2. Validates core CPU computer-vision operations (Canny, Sobel, Blur, etc.).
3. Logs system architecture, CPU core count, and OpenCV build flags to docs/environment_verification.json.
"""

import sys
import platform
import os
import json
import time

def verify_opencv5():
    print("=" * 70)
    print("VISIONLOOP: OPENCV 5 ENVIRONMENT VERIFICATION (STOP-THE-LINE CHECK)")
    print("=" * 70)

    # 1. Check Python version
    py_ver = sys.version.split()[0]
    print(f"[1/5] Checking Python version... {py_ver}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        print(f"ERROR: Python 3.10+ required. Found {py_ver}")
        sys.exit(1)
    print("      -> Python version satisfies requirement (>= 3.10).")

    # 2. Check cv2 import and version string
    print("[2/5] Importing cv2 and checking version...")
    try:
        import cv2
    except ImportError as e:
        print(f"CRITICAL ERROR: Failed to import cv2: {e}")
        sys.exit(1)

    cv_version = cv2.__version__
    print(f"      -> Detected OpenCV version: {cv_version}")
    if not cv_version.startswith("5."):
        print(f"CRITICAL ERROR: OpenCV 5 is required by competition rules! Found: {cv_version}")
        sys.exit(1)
    print("      -> PASS: Verified OpenCV 5 runtime.")

    # 3. Check CPU Hardware & Threading Settings
    print("[3/5] Verifying CPU acceleration & thread configuration...")
    import numpy as np

    threads = cv2.getNumThreads()
    optimized = cv2.useOptimized()
    cpu_cores = os.cpu_count() or 1
    arch = platform.machine()
    system_os = f"{platform.system()} {platform.release()}"

    print(f"      -> System OS: {system_os} ({arch})")
    print(f"      -> Logical CPU Cores: {cpu_cores}")
    print(f"      -> OpenCV Threads: {threads}")
    print(f"      -> OpenCV CPU Optimization Active: {optimized}")

    # 4. Verify Essential Computer Vision Operations
    print("[4/5] Testing core CPU vision operations with synthetic data...")

    # Create synthetic test pattern with sharp edges
    h, w = 256, 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (216, 216), (255, 255, 255), -1)
    cv2.circle(img, (128, 128), 50, (0, 0, 0), -1)

    t0 = time.perf_counter()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (h, w), "Grayscale conversion shape mismatch"

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    assert blurred.shape == (h, w), "Gaussian blur shape mismatch"

    # Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3, L2gradient=True)
    assert edges.shape == (h, w), "Canny output shape mismatch"
    edge_pixel_count = int(np.count_nonzero(edges))
    assert edge_pixel_count > 0, "Canny edge detection failed to find synthetic edges"

    # Sobel Gradient Filter
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    assert sobel_mag.shape == (h, w), "Sobel gradient magnitude mismatch"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"      -> All core vision functions passed in {elapsed_ms:.2f} ms.")
    print(f"      -> Detected {edge_pixel_count} edge pixels in synthetic geometry.")

    # 5. Export Environment Verification Artifact
    print("[5/5] Writing environment verification report...")
    build_info = cv2.getBuildInformation()
    
    report = {
        "status": "VERIFIED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": py_ver,
        "opencv_version": cv_version,
        "platform": {
            "os": system_os,
            "architecture": arch,
            "cpu_cores": cpu_cores,
            "processor": platform.processor(),
        },
        "opencv_cpu_config": {
            "num_threads": threads,
            "use_optimized": optimized,
        },
        "smoke_test": {
            "synthetic_image_dimensions": [h, w],
            "detected_edge_pixels": edge_pixel_count,
            "execution_latency_ms": round(elapsed_ms, 3),
        },
        "opencv_build_summary": build_info[:500] + "..."
    }

    os.makedirs("docs", exist_ok=True)
    report_path = os.path.join("docs", "environment_verification.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"      -> Verification report saved to: {report_path}")
    print("=" * 70)
    print("PHASE 0 STOP-THE-LINE CHECK: SUCCESSFUL! OPENCV 5 RUNTIME CONFIRMED.")
    print("=" * 70)

if __name__ == "__main__":
    verify_opencv5()
