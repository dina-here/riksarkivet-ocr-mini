import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def preprocess_mild(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mild contrast boost
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Very light denoise
    gray = cv2.fastNlMeansDenoising(gray, h=5)

    return gray

def preprocess_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in input_dir.glob("*"):
        img = cv2.imread(str(p))
        out = preprocess_mild(img)
        out_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(out_path), out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    preprocess_folder(args.input_dir, args.output_dir)
