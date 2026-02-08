from __future__ import annotations

import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def preprocess_image_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess for OCR:
    - grayscale
    - CLAHE contrast
    - denoise
    - adaptive threshold
    - morphology cleanup
    Returns a single-channel uint8 image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement (good for scans)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Mild denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold (robust to uneven lighting)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    # Remove small speckles / close tiny gaps
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    return thr


def preprocess_folder(input_dir: str, output_dir: str) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]

    if not paths:
        raise FileNotFoundError(f"No images found in {in_dir}")

    for p in tqdm(paths, desc="Preprocessing"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p}")
            continue

        out = preprocess_image_bgr(img)

        rel = p.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as PNG to avoid JPEG artifacts
        out_path = out_path.with_suffix(".png")
        cv2.imwrite(str(out_path), out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    preprocess_folder(args.input_dir, args.output_dir)
