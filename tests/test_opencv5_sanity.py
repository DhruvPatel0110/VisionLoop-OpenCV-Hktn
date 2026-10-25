"""Sanity test verifying that OpenCV 5 runs an end-to-end CPU edge detection pipeline.

Can be run directly via `python tests/test_opencv5_sanity.py` or with `pytest`.
"""

import os
import time
import cv2
import numpy as np

def run_sanity_pipeline():
    print("Running OpenCV 5 Sanity Pipeline...")
    
    # Verify input exists
    input_path = "data/test_images/sample_01.png"
    assert os.path.exists(input_path), f"Input image not found: {input_path}"
    
    # 1. Read input image
    t_start = time.perf_counter()
    img = cv2.imread(input_path)
    assert img is not None, f"Failed to load image from {input_path}"
    h, w, c = img.shape
    assert h == 512 and w == 512 and c == 3, f"Unexpected image shape: {img.shape}"
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Gaussian filter (OpenCV 5 CPU)
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
    
    # 4. Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3, L2gradient=True)
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    
    # 5. Assertions on output
    assert edges.shape == (h, w), "Output dimension mismatch"
    assert edges.dtype == np.uint8, f"Output dtype mismatch: {edges.dtype}"
    edge_pixel_count = int(np.count_nonzero(edges))
    assert edge_pixel_count > 500, f"Expected > 500 edge pixels, got {edge_pixel_count}"
    
    # 6. Save output
    output_path = "data/sanity_output.png"
    success = cv2.imwrite(output_path, edges)
    assert success, f"Failed to write output image to {output_path}"
    assert os.path.exists(output_path), "Saved output file does not exist"
    
    print(f"Sanity Pipeline SUCCESS:")
    print(f"  Input: {input_path} ({w}x{h})")
    print(f"  Output: {output_path}")
    print(f"  Detected Edges: {edge_pixel_count} pixels")
    print(f"  Execution Time: {t_elapsed_ms:.2f} ms")
    return True

def test_opencv5_sanity():
    """Pytest test case."""
    assert run_sanity_pipeline() is True

if __name__ == "__main__":
    run_sanity_pipeline()
