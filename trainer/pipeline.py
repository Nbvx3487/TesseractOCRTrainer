"""Fine-tune Tesseract's English LSTM model on a custom font.

The steps mirror Tesseract's own training workflow (tesstrain):
  1. Extract the base model's LSTM network from eng.traineddata.
  2. Convert each training image + .gt.txt into a .lstmf file.
  3. Continue training from the base network on those files.
  4. Stop training and package the result as a .traineddata file.

Requires the Tesseract training tools (lstmtraining, combine_tessdata)
and the base language data (eng.traineddata).
"""

import os
import shutil
import subprocess
from pathlib import Path

from .corpus import split_corpus
from .render import render_lines
from .scoring import score_set


def find_system_tessdata():
    """Locate the directory holding eng.traineddata, or raise."""
    candidates = []
    if os.environ.get("TESSDATA_PREFIX"):
        candidates.append(Path(os.environ["TESSDATA_PREFIX"]))
    candidates += [
        Path("/usr/share/tesseract-ocr/5/tessdata"),
        Path("/usr/share/tesseract-ocr/4.00/tessdata"),
        Path("/usr/share/tessdata"),
        Path("C:/Program Files/Tesseract-OCR/tessdata"),
    ]
    for directory in candidates:
        if (directory / "eng.traineddata").exists():
            return directory
    raise FileNotFoundError("eng.traineddata not found. Set TESSDATA_PREFIX to its folder.")


BEST_MODEL_HINT = (
    "Fine-tuning needs the 'best' English model (eng.traineddata from "
    "https://github.com/tesseract-ocr/tessdata_best). Put it in the base_models folder "
    "next to app.py, or set TESSDATA_BEST_DIR to the folder that contains it."
)


def find_best_model():
    """Locate the floating-point 'best' eng.traineddata that can be fine-tuned."""
    candidates = []
    if os.environ.get("TESSDATA_BEST_DIR"):
        candidates.append(Path(os.environ["TESSDATA_BEST_DIR"]))
    candidates.append(Path(__file__).resolve().parent.parent / "base_models")
    for directory in candidates:
        model = directory / "eng.traineddata"
        if model.exists() and model.stat().st_size > 1_000_000:
            return model
    raise FileNotFoundError(BEST_MODEL_HINT)


def run(cmd, log, cwd=None):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", cwd=cwd)
    if result.stdout.strip():
        log(result.stdout.strip()[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr.strip()[-2000:]}")
    return result


def train_font(name, font_path, corpus, control_fonts, data_root, log,
               font_size=32, iterations=400):
    """Run the full pipeline. Returns a dict of scores and output paths."""
    work = Path(data_root) / name
    if work.exists():
        shutil.rmtree(work)
    base_dir = work / "base"
    base_dir.mkdir(parents=True)

    system_tessdata = find_system_tessdata()
    shutil.copy(find_best_model(), base_dir / "eng.traineddata")
    run(["combine_tessdata", "-e", str(base_dir / "eng.traineddata"),
         str(base_dir / "eng.lstm")], log)

    train_lines, control_lines, dropped = split_corpus(corpus)
    log(f"Corpus: {len(train_lines)} training lines, {len(control_lines)} control lines")
    if dropped:
        log(f"Skipped {dropped} Cyrillic line(s): the English model cannot learn them.")

    train_dir = work / "train"
    render_lines(train_lines, font_path, train_dir, font_size, prefix="train")

    lstmf_list = []
    for tif in sorted(train_dir.glob("*.tif")):
        run(["tesseract", str(tif), str(tif.with_suffix("")),
             "--psm", "13", "lstm.train"], log)
        lstmf_list.append(str(tif.with_suffix(".lstmf")))
    list_file = work / "train_list.txt"
    list_file.write_text("\n".join(lstmf_list) + "\n", encoding="utf-8")

    checkpoints = work / "checkpoints"
    checkpoints.mkdir()
    run(["lstmtraining",
         "--model_output", str(checkpoints / name),
         "--continue_from", str(base_dir / "eng.lstm"),
         "--traineddata", str(base_dir / "eng.traineddata"),
         "--train_listfile", str(list_file),
         "--max_iterations", str(iterations)], log)

    output_dir = work / "model"
    output_dir.mkdir()
    run(["lstmtraining", "--stop_training",
         "--continue_from", str(checkpoints / f"{name}_checkpoint"),
         "--traineddata", str(base_dir / "eng.traineddata"),
         "--model_output", str(output_dir / f"{name}.traineddata")], log)

    # Control set: held-out lines in the trained font plus any extra fonts,
    # so forgetting on other fonts shows up in the numbers.
    control_images = {}
    fonts = [("trained_font", font_path)] + [
        (Path(f).stem, f) for f in control_fonts
    ]
    for label, path in fonts:
        control_images[label] = render_lines(
            control_lines, path, work / "control" / label, font_size, prefix="ctrl")

    results = {"name": name, "model": str(output_dir / f"{name}.traineddata"), "scores": {}}
    for label, images in control_images.items():
        # Scoring reads .tif the same as .png, so the same pipeline covers both.
        results["scores"][label] = {
            "baseline": score_set(images, "eng", system_tessdata),
            "trained": score_set(images, name, output_dir),
        }
        log(f"{label}: baseline {results['scores'][label]['baseline']['exact']}/"
            f"{len(images)} exact, trained {results['scores'][label]['trained']['exact']}/"
            f"{len(images)} exact")
    return results
