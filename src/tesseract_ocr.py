from __future__ import annotations
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import os
import pytesseract

# Använd samma tesseract.exe som du testade i PowerShell:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Använd kort sökväg för tessdata (undviker spaces + quote-problem)
TESSDATA_DIR = r"C:\PROGRA~1\Tesseract-OCR\tessdata"

def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


def ocr_folder(
    input_dir: str,
    output_dir: str,
    lang: str = "swe+eng",
    psm: int = 6,
    oem: int = 3,
) -> None:
    """
    psm (Page Segmentation Mode):
      3 = auto (hela sidan)
      4 = single column
      6 = single uniform block of text (bra default)
      11 = sparse text (bra för kartor/labels ibland)
    oem (OCR Engine Mode):
      3 = default (LSTM + legacy om tillgängligt)
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(in_dir)

    if not paths:
        raise FileNotFoundError(f"Inga bilder hittades i {in_dir}")

    config = f"--oem {oem} --psm {psm} --tessdata-dir {TESSDATA_DIR}"

    for p in tqdm(paths, desc=f"Tesseract OCR ({lang}, psm={psm})"):
        try:
            img = Image.open(p)
        except Exception as e:
            print(f"Varning: kunde inte öppna {p}: {e}")
            continue

        text = pytesseract.image_to_string(img, lang=lang, config=config).strip()

        rel = p.relative_to(in_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path = out_path.with_suffix(".txt")
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lang", default="swe+eng")
    ap.add_argument("--psm", type=int, default=6)
    ap.add_argument("--oem", type=int, default=3)
    args = ap.parse_args()

    ocr_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lang=args.lang,
        psm=args.psm,
        oem=args.oem,
    )
