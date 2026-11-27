"""Render text lines into TIFF images with box files, using Tesseract's text2image.

Tesseract's LSTM trainer needs, per line:
  name.tif     the rendered line
  name.box     character boxes (written by text2image)
  name.gt.txt  the exact text (used for scoring)
"""

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import ImageFont

# Per-user font folder on Linux. fontconfig scans it by default, so text2image finds
# any font copied here. Fonts on a Windows drive (/mnt/c/...) are not scanned.
USER_FONTS = Path.home() / ".local" / "share" / "fonts"


def font_family(font_path):
    """Return the family name Pango/text2image uses for this font file."""
    family, _style = ImageFont.truetype(str(font_path), 12).getname()
    return family


def stage_font(font_path):
    """Copy the font into the per-user font folder and rebuild that folder's font cache."""
    USER_FONTS.mkdir(parents=True, exist_ok=True)
    source = Path(font_path)
    target = USER_FONTS / source.name
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target)
        subprocess.run(["fc-cache", "-f", str(USER_FONTS)], check=True,
                       capture_output=True)
    return USER_FONTS


def render_one(text, stem, family, fonts_dir, font_size):
    """Render one line with text2image and write its .gt.txt. Returns the TIFF path."""
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
    return stem.with_suffix(".tif")


def render_lines(lines, font_path, out_dir, font_size=32, prefix="line", log=None, label="",
                 progress=None):
    """Render each line to name.tif + name.box + name.gt.txt. Returns the TIFF paths.

    Lines are rendered in parallel, one per CPU thread (OMP_THREAD_LIMIT, set by the
    window). If log is given, one line is logged per rendered image. If progress is
    given, it is called as progress(done, total) after each image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    family = font_family(font_path)
    fonts_dir = stage_font(font_path)
    workers = max(1, int(os.environ.get("OMP_THREAD_LIMIT", "4")))

    tiffs = [None] * len(lines)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for index, text in enumerate(lines):
            stem = out_dir / f"{prefix}_{index:04d}"
            futures[pool.submit(render_one, text, stem, family, fonts_dir, font_size)] = index
        try:
            for future in as_completed(futures):
                tiffs[futures[future]] = future.result()
                done += 1
                if log:
                    log(f"  rendered {done}/{len(lines)} lines in {label}")
                if progress:
                    progress(done, len(lines))
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return tiffs
