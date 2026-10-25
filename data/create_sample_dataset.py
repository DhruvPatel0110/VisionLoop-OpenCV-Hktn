"""Dataset generator for benchmark test images and ground-truth edge maps.

Generates 5 diverse static test images (max 512x512) along with exact reference edge maps:
  1. sample_01.png: High-contrast geometric shapes with clean boundaries.
  2. sample_02.png: Smooth contrast gradient with circular features.
  3. sample_03_noisy.png: High-contrast badge over a noisy/textured background.
  4. sample_04_low_contrast.png: Low-light / low-contrast subtle boundaries.
  5. sample_05_fine_detail.png: Fine concentric circles and grid lines.
"""

import os
import json
import cv2
import numpy as np

def generate_full_dataset():
    img_dir = "data/test_images"
    ref_dir = "data/reference_edges"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    h, w = 512, 512
    manifest = []

    # -------------------------------------------------------------
    # 1. sample_01.png: High contrast geometric shapes
    # -------------------------------------------------------------
    img1 = np.ones((h, w, 3), dtype=np.uint8) * 240
    ref1 = np.zeros((h, w), dtype=np.uint8)

    cv2.rectangle(img1, (30, 30), (482, 482), (40, 40, 40), 4)
    cv2.rectangle(ref1, (30, 30), (482, 482), 255, 1)

    cv2.rectangle(img1, (70, 70), (220, 220), (30, 100, 200), -1)
    cv2.rectangle(ref1, (70, 70), (220, 220), 255, 1)

    cv2.circle(img1, (360, 150), 75, (20, 160, 40), -1)
    cv2.circle(ref1, (360, 150), 75, 255, 1)

    pts1 = np.array([[150, 430], [280, 270], [420, 430]], np.int32)
    cv2.polylines(img1, [pts1], isClosed=True, color=(180, 40, 180), thickness=4)
    cv2.fillPoly(img1, [pts1], color=(210, 100, 210))
    cv2.polylines(ref1, [pts1], isClosed=True, color=255, thickness=1)

    p1_img = os.path.join(img_dir, "sample_01.png").replace("\\", "/")
    p1_ref = os.path.join(ref_dir, "sample_01_ref.png").replace("\\", "/")
    cv2.imwrite(p1_img, img1)
    cv2.imwrite(p1_ref, ref1)
    manifest.append({
        "id": "sample_01",
        "name": "High-Contrast Geometric Shapes",
        "image_path": p1_img,
        "reference_path": p1_ref,
        "dimensions": [w, h],
        "category": "geometric_clean",
        "has_ground_truth": True
    })

    # -------------------------------------------------------------
    # 2. sample_02.png: Smooth contrast gradient with circular features
    # -------------------------------------------------------------
    img2 = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img2[y, :, :] = int(y / h * 190)

    cv2.circle(img2, (160, 200), 90, (230, 230, 230), -1)
    cv2.circle(img2, (350, 320), 110, (130, 130, 130), -1)

    ref2 = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ref2, (160, 200), 90, 255, 1)
    cv2.circle(ref2, (350, 320), 110, 255, 1)

    p2_img = os.path.join(img_dir, "sample_02.png").replace("\\", "/")
    p2_ref = os.path.join(ref_dir, "sample_02_ref.png").replace("\\", "/")
    cv2.imwrite(p2_img, img2)
    cv2.imwrite(p2_ref, ref2)
    manifest.append({
        "id": "sample_02",
        "name": "Smooth Gradient Circles",
        "image_path": p2_img,
        "reference_path": p2_ref,
        "dimensions": [w, h],
        "category": "soft_gradient",
        "has_ground_truth": True
    })

    # -------------------------------------------------------------
    # 3. sample_03_noisy.png: High-contrast badge over textured noise
    # -------------------------------------------------------------
    np.random.seed(42)
    noise = np.random.randint(60, 110, (h, w), dtype=np.uint8)
    img3 = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    ref3 = np.zeros((h, w), dtype=np.uint8)

    # Sharp central badge that should be extracted despite background noise
    cv2.rectangle(img3, (120, 140), (392, 372), (250, 250, 250), -1)
    cv2.rectangle(img3, (130, 150), (382, 362), (20, 20, 20), -1)
    cv2.putText(img3, "OPENCV 5", (155, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    cv2.rectangle(ref3, (120, 140), (392, 372), 255, 1)
    cv2.rectangle(ref3, (130, 150), (382, 362), 255, 1)

    p3_img = os.path.join(img_dir, "sample_03_noisy.png").replace("\\", "/")
    p3_ref = os.path.join(ref_dir, "sample_03_noisy_ref.png").replace("\\", "/")
    cv2.imwrite(p3_img, img3)
    cv2.imwrite(p3_ref, ref3)
    manifest.append({
        "id": "sample_03_noisy",
        "name": "Textured Noise with Central Badge",
        "image_path": p3_img,
        "reference_path": p3_ref,
        "dimensions": [w, h],
        "category": "noisy_textured",
        "has_ground_truth": True
    })

    # -------------------------------------------------------------
    # 4. sample_04_low_contrast.png: Low-light / subtle contrast
    # -------------------------------------------------------------
    img4 = np.ones((h, w, 3), dtype=np.uint8) * 35  # dark background
    ref4 = np.zeros((h, w), dtype=np.uint8)

    # Subtle contrast rectangles (+30 intensity)
    cv2.rectangle(img4, (80, 80), (240, 420), (65, 65, 65), -1)
    cv2.rectangle(ref4, (80, 80), (240, 420), 255, 1)

    # Subtle contrast circle (+25 intensity)
    cv2.circle(img4, (370, 256), 100, (60, 60, 60), -1)
    cv2.circle(ref4, (370, 256), 100, 255, 1)

    p4_img = os.path.join(img_dir, "sample_04_low_contrast.png").replace("\\", "/")
    p4_ref = os.path.join(ref_dir, "sample_04_low_contrast_ref.png").replace("\\", "/")
    cv2.imwrite(p4_img, img4)
    cv2.imwrite(p4_ref, ref4)
    manifest.append({
        "id": "sample_04_low_contrast",
        "name": "Low-Light Subtle Boundaries",
        "image_path": p4_img,
        "reference_path": p4_ref,
        "dimensions": [w, h],
        "category": "low_contrast",
        "has_ground_truth": True
    })

    # -------------------------------------------------------------
    # 5. sample_05_fine_detail.png: Fine grid lines and concentric circles
    # -------------------------------------------------------------
    img5 = np.ones((h, w, 3), dtype=np.uint8) * 255
    ref5 = np.zeros((h, w), dtype=np.uint8)

    # Fine grid lines (tests over-smoothing vs preservation)
    for coord in range(60, 460, 50):
        cv2.line(img5, (coord, 60), (coord, 450), (30, 30, 30), 1)
        cv2.line(ref5, (coord, 60), (coord, 450), 255, 1)
        cv2.line(img5, (60, coord), (450, coord), (30, 30, 30), 1)
        cv2.line(ref5, (60, coord), (450, coord), 255, 1)

    # Concentric rings
    for r in (40, 80, 120):
        cv2.circle(img5, (256, 256), r, (10, 10, 10), 1)
        cv2.circle(ref5, (256, 256), r, 255, 1)

    p5_img = os.path.join(img_dir, "sample_05_fine_detail.png").replace("\\", "/")
    p5_ref = os.path.join(ref_dir, "sample_05_fine_detail_ref.png").replace("\\", "/")
    cv2.imwrite(p5_img, img5)
    cv2.imwrite(p5_ref, ref5)
    manifest.append({
        "id": "sample_05_fine_detail",
        "name": "Fine Grid and Concentric Rings",
        "image_path": p5_img,
        "reference_path": p5_ref,
        "dimensions": [w, h],
        "category": "fine_detail",
        "has_ground_truth": True
    })

    # Save manifest
    manifest_path = "data/dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated {len(manifest)} curated benchmark images with ground truth!")
    print(f"Dataset manifest written to {manifest_path}")

if __name__ == "__main__":
    generate_full_dataset()
