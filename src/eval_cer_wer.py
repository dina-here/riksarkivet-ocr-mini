from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from jiwer import wer, cer


@dataclass
class Metrics:
    n: int
    wer: float
    cer: float


def evaluate(ground_truth_dir: str, pred_dir: str) -> Metrics:
    gt_dir = Path(ground_truth_dir)
    pr_dir = Path(pred_dir)

    gt_files = sorted(gt_dir.rglob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No ground truth .txt files found in {gt_dir}")

    wers = []
    cers = []
    matched = 0

    for gt_path in gt_files:
        rel = gt_path.relative_to(gt_dir)
        pred_path = pr_dir / rel

        if not pred_path.exists():
            # Skip if missing prediction
            continue

        gt = gt_path.read_text(encoding="utf-8").strip()
        pr = pred_path.read_text(encoding="utf-8").strip()

        # Avoid crashing on empty strings
        if len(gt) == 0:
            continue

        wers.append(wer(gt, pr))
        cers.append(cer(gt, pr))
        matched += 1

    if matched == 0:
        raise FileNotFoundError(
            f"No matching prediction files found in {pr_dir} for ground truth in {gt_dir}"
        )

    return Metrics(n=matched, wer=sum(wers) / matched, cer=sum(cers) / matched)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True, help="Folder with ground truth .txt files")
    ap.add_argument("--pred_dir", required=True, help="Folder with OCR prediction .txt files")
    args = ap.parse_args()

    m = evaluate(args.gt_dir, args.pred_dir)
    print(f"Matched files: {m.n}")
    print(f"WER: {m.wer:.4f}")
    print(f"CER: {m.cer:.4f}")
