import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def remove_form_lines(gray):
    # binarisera (inverterad: text/linjer = vitt)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 35, 15)

    # hitta horisontella linjer
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, h_kernel, iterations=1)

    # hitta vertikala linjer
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    v_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, v_kernel, iterations=1)

    lines = cv2.bitwise_or(h_lines, v_lines)

    # “måla bort” linjer från binär bild
    thr_no_lines = cv2.bitwise_and(thr, cv2.bitwise_not(lines))

    # tillbaka till normal (svart text på vit bakgrund)
    out = cv2.bitwise_not(thr_no_lines)
    return out

def process_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
    paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images in {input_dir}")

    for p in tqdm(paths, desc="Remove lines"):
        img = cv2.imread(str(p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        out = remove_form_lines(gray)

        out_path = output_dir / p.with_suffix(".png").name
        cv2.imwrite(str(out_path), out_path=out_path.as_posix())  # safety
        cv2.imwrite(str(out_path), out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    process_folder(args.input_dir, args.output_dir)
