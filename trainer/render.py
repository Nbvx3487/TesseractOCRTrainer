"""Render text lines into TIFF images with box files, using Tesseract's text2image.

Tesseract's LSTM trainer needs, per line:
  name.tif     the rendered line
  name.box     character boxes (written by text2image)
  name.gt.txt  the exact text (used for scoring)
"""

import subprocess
from pathlib import Path

from PIL import ImageFont


def font_family(font_path):
    """Return the family name Pango/text2image uses for this font file."""
    family, _style = ImageFont.truetype(str(font_path), 12).getname()
    return family


def render_lines(lines, font_path, out_dir, font_size=32, prefix="line"):
    """Render each line to name.tif + name.box + name.gt.txt. Returns the TIFF paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    family = font_family(font_path)
    fonts_dir = Path(font_path).resolve().parent

    tiffs = []
    for index, text in enumerate(lines):
        stem = out_dir / f"{prefix}_{index:04d}"
        text_file = stem.with_suffix(".txt")
        text_file.write_text(text + "\n", encoding="utf-8")
        result = subprocess.run(
            ["text2image", f"--text={text_file}", f"--outputbase={stem}",
             f"--font={family}", f"--fonts_dir={fonts_dir}", f"--ptsize={font_size}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0 or not stem.with_suffix(".tif").exists():
            raise RuntimeError(f"text2image failed for '{text}': {result.stderr.strip()}")
        stem.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")
        text_file.unlink()
        tiffs.append(stem.with_suffix(".tif"))
    return tiffs
