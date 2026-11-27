"""Fine-tune Tesseract's English LSTM model on one or more fonts.

Each font in the dataset has a weight: the number of times its training images
are repeated in the training list, so a weight of 3 means the model sees that
font three times as often. Every font is also scored separately on held-out
lines, so forgetting on any font shows up in the results.

Steps:
  1. Extract the base model's LSTM network from the best eng.traineddata.
  2. Render each training line in each font with text2image.
  3. Convert each image into a .lstmf file, and list each one weight times.
  4. Continue training from the base network on that list.
  5. Stop training and package the result as a .traineddata file.
  6. Score the base and trained models on held-out lines, per font.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .corpus import split_corpus
from .render import render_lines
from .scoring import score_set

BEST_MODEL_HINT = (
    "Fine-tuning needs the 'best' English model (eng.traineddata from "
    "https://github.com/tesseract-ocr/tessdata_best). Put it in the base_models folder "
    "next to app.py, or set TESSDATA_BEST_DIR to the folder that contains it."
)


def find_system_tessdata():
    """Locate the directory holding the installed eng.traineddata, or raise."""
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


ITERATION_LINE = re.compile(r"At iteration (\d+)/")


def run(cmd, log, quiet=False, on_line=None):
    """Run a command and stream its output line by line, so the log keeps moving.
    quiet=True skips logging the command and its lines (used for per-line steps)."""
    if not quiet:
        log(f"$ {' '.join(str(c) for c in cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    output = []
    for raw in process.stdout:
        line = raw.rstrip()
        if not line:
            continue
        output.append(line)
        if on_line:
            on_line(line)
        if not quiet:
            log(line)
    code = process.wait()
    if code != 0:
        raise RuntimeError(f"{cmd[0]} failed: {' | '.join(output[-5:])[-2000:]}")


def report(log, stage, label, current, count, done, total):
    """A progress line for the window. The window reads these and does not log them."""
    log("PROGRESS:" + json.dumps({"stage": stage, "label": label, "current": current,
                                  "count": count, "done": done, "total": total}))


def font_label(index, font_path):
    return f"{index}_{Path(font_path).stem}"


def train_font(name, fonts, corpus, data_root, log, font_size=32, iterations=400):
    """Run the full pipeline.

    fonts: list of (font_path, weight) pairs. Weight is an integer, 1 or more.
    Returns a dict with per-font scores and the path to the trained model.
    """
    if not fonts:
        raise ValueError("Add at least one font to the dataset.")
    for _, weight in fonts:
        if weight < 1:
            raise ValueError("Each font weight must be 1 or more.")

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
    log(f"Corpus: {len(train_lines)} training lines, {len(control_lines)} held-out lines")
    if dropped:
        log(f"Skipped {dropped} Cyrillic line(s): the English model cannot learn them.")

    # Progress units: one per training line prepared, one per training iteration,
    # and two per held-out line scored (before and after) in each font.
    font_count = len(fonts)
    prepared_units = font_count * len(train_lines)
    total = prepared_units + iterations + font_count * len(control_lines) * 2
    done = 0

    # Render training images per font, and list each one weight times.
    lstmf_list = []
    for index, (font_path, weight) in enumerate(fonts):
        label = font_label(index, font_path)
        train_dir = work / "train" / label
        log(f"Rendering {len(train_lines)} training lines in {label} (weight {weight})")
        tifs = render_lines(train_lines, font_path, train_dir, font_size, prefix="train")
        for number, tif in enumerate(tifs, start=1):
            run(["tesseract", str(tif), str(tif.with_suffix("")),
                 "--psm", "13", "lstm.train"], log, quiet=True)
            entry = str(tif.with_suffix(".lstmf"))
            lstmf_list.extend([entry] * weight)
            done += 1
            report(log, "prepare", label, number, len(tifs), done, total)
        log(f"Prepared {len(tifs)} lines in {label}")

    list_file = work / "train_list.txt"
    list_file.write_text("\n".join(lstmf_list) + "\n", encoding="utf-8")
    log(f"Training list: {len(lstmf_list)} entries")

    checkpoints = work / "checkpoints"
    checkpoints.mkdir()

    def on_training_line(line):
        match = ITERATION_LINE.search(line)
        if match:
            current = min(int(match.group(1)), iterations)
            report(log, "train", "all fonts", current, iterations,
                   prepared_units + current, total)

    report(log, "train", "all fonts", 0, iterations, prepared_units, total)
    run(["lstmtraining",
         "--model_output", str(checkpoints / name),
         "--continue_from", str(base_dir / "eng.lstm"),
         "--traineddata", str(base_dir / "eng.traineddata"),
         "--train_listfile", str(list_file),
         "--max_iterations", str(iterations)], log, on_line=on_training_line)
    done = prepared_units + iterations

    output_dir = work / "model"
    output_dir.mkdir()
    run(["lstmtraining", "--stop_training",
         "--continue_from", str(checkpoints / f"{name}_checkpoint"),
         "--traineddata", str(base_dir / "eng.traineddata"),
         "--model_output", str(output_dir / f"{name}.traineddata")], log)

    # Score both models on the held-out lines in every dataset font.
    results = {"name": name, "model": str(output_dir / f"{name}.traineddata"), "scores": {}}
    for index, (font_path, _) in enumerate(fonts):
        label = font_label(index, font_path)
        log(f"Rendering {len(control_lines)} held-out lines in {label}")
        images = render_lines(control_lines, font_path, work / "control" / label,
                              font_size, prefix="ctrl")
        report(log, "score", label, 0, len(images), done, total)
        log(f"Scoring {label}: before training, then after")
        baseline = score_set(images, "eng", system_tessdata)
        trained = score_set(images, name, output_dir)
        done += len(images) * 2
        report(log, "score", label, len(images), len(images), done, total)
        results["scores"][label] = {"baseline": baseline, "trained": trained}
        log(f"{label}: before {baseline['exact']}/{len(images)} exact, "
            f"after {trained['exact']}/{len(images)} exact")
    return results
