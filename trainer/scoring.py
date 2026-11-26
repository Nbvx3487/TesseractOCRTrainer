"""Measure OCR accuracy on a control set: exact-match rate and character error rate."""

import subprocess
from pathlib import Path


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def run_tesseract(image_path, lang, tessdata_dir):
    result = subprocess.run(
        ["tesseract", str(image_path), "stdout", "-l", lang,
         "--tessdata-dir", str(tessdata_dir), "--psm", "7"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"tesseract failed on {image_path}: {result.stderr.strip()}")
    return " ".join(result.stdout.split())


def score_set(image_paths, lang, tessdata_dir):
    """Return a dict with exact-match count, total, mean CER, and the first few misses."""
    exact = 0
    total_chars = 0
    total_errors = 0
    misses = []
    for image in image_paths:
        image = Path(image)
        truth = image.with_suffix(".gt.txt").read_text(encoding="utf-8").strip()
        got = run_tesseract(image, lang, tessdata_dir)
        distance = edit_distance(got, truth)
        total_errors += distance
        total_chars += max(len(truth), 1)
        if got == truth:
            exact += 1
        elif len(misses) < 5:
            misses.append({"expected": truth, "got": got})
    return {
        "exact": exact,
        "total": len(image_paths),
        "cer": round(total_errors / max(total_chars, 1), 4),
        "misses": misses,
    }
