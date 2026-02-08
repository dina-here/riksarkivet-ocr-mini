from __future__ import annotations

from pathlib import Path
from typing import Iterable
from PIL import Image
from tqdm import tqdm

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]


@torch.inference_mode()
def ocr_folder(
    input_dir: str,
    output_dir: str,
    model_name: str = "microsoft/trocr-base-printed",
    max_new_tokens: int = 128,
) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    model.eval()

    paths = list_images(in_dir)
    if not paths:
        raise FileNotFoundError(f"No images found in {in_dir}")

    for p in tqdm(paths, desc=f"OCR ({model_name})"):
        try:
            image = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Warning: could not open {p}: {e}")
            continue

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=4,            # quality vs speed
            early_stopping=True,
        )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
    ap.add_argument("--model_name", default="microsoft/trocr-base-printed")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    ocr_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
    )
