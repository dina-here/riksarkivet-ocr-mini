from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def preprocess_printed(img_bgr: np.ndarray) -> np.ndarray:
    """
    För tryckta dokument/formulär: robust binarisering + lite cleanup.
    Returnerar en 1-kanals uint8-bild.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Kontrast (bra för bleka skanningar)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise lite
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold: funkar ofta bra på skanningar
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    # Liten morfologi för att ta bort småprickar
    kernel = np.ones((2, 2), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    return thr


def preprocess_handwritten(img_bgr: np.ndarray) -> np.ndarray:
    """
    För handskrift: mildare preprocessing (undvik hård binarisering som kan äta upp tunna streck).
    Returnerar en 1-kanals uint8-bild.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Lätt blur kan hjälpa mot brus utan att döda streck
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray


def preprocess_folder(input_dir: str, output_dir: str, profile: str) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    paths = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]

    if not paths:
        raise FileNotFoundError(f"Inga bilder hittades i {in_dir}")

    for p in tqdm(paths, desc=f"Preprocess ({profile})"):
        img = cv2.imread(str(p))
        if img is None:
            print(f"Varning: kunde inte läsa {p}")
            continue

        if profile == "printed":
            out = preprocess_printed(img)
        elif profile == "handwritten":
            out = preprocess_handwritten(img)
        else:
            raise ValueError("profile måste vara 'printed' eller 'handwritten'")

        rel = p.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path.with_suffix(".png")  # spara alltid png
        cv2.imwrite(str(out_path), out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--profile", choices=["printed", "handwritten"], required=True)
    args = ap.parse_args()

    preprocess_folder(args.input_dir, args.output_dir, args.profile)
