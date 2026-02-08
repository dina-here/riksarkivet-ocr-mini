# src/segment_lines.py
import cv2
import numpy as np
from pathlib import Path

def segment_lines(img_path: str, out_dir: str, crop_left_frac: float = 0.28):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop bort ornament till vänster
    h, w = gray.shape
    x0 = int(w * crop_left_frac)
    gray = gray[:, x0:]

    # Mild binarization för line finding (inte för OCR)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 15)

    # Koppla ihop bokstäver till "rader"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 3))
    connected = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortera rader uppifrån och ned
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted(boxes, key=lambda b: b[1])

    for i, (x, y, ww, hh) in enumerate(boxes):
        if hh < 12 or ww < 80:
            continue
        pad = 8
        y1 = max(0, y - pad)
        y2 = min(gray.shape[0], y + hh + pad)
        x1 = max(0, x - pad)
        x2 = min(gray.shape[1], x + ww + pad)
        line = gray[y1:y2, x1:x2]
        cv2.imwrite(str(out_dir / f"line_{i:03d}.png"), line)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--crop_left_frac", type=float, default=0.28)
    args = ap.parse_args()
    segment_lines(args.img, args.out_dir, args.crop_left_frac)
