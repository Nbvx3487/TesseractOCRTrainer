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

import os
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


def run(cmd, log):
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace")
    if result.stdout.strip():
        log(result.stdout.strip()[-2000:])
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr.strip()[-2000:]}")
    return result


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

    corpus_text = corpus
    train_lines, control_lines, dropped = split_corpus(corpus_text)
    log(f"Corpus: {len(train_lines)} training lines, {len(control_lines)} held-out lines")
    if dropped:
        log(f"Skipped {dropped} Cyrillic line(s): the English model cannot learn them.")

    # Render training images per font, and list each one weight times.
    lstmf_list = []
    for index, (font_path, weight) in enumerate(fonts):
        label = font_label(index, font_path)
        train_dir = work / "train" / label
        tifs = render_lines(train_lines, font_path, train_dir, font_size, prefix="train")
        log(f"Rendered {len(tifs)} lines in {label} (weight {weight})")
        for tif in tifs:
            run(["tesseract", str(tif), str(tif.with_suffix("")),
                 "--psm", "13", "lstm.train"], log)
            entry = str(tif.with_suffix(".lstmf"))
            lstmf_list.extend([entry] * weight)

    list_file = work / "train_list.txt"
    list_file.write_text("\n".join(lstmf_list) + "\n", encoding="utf-8")
    log(f"Training list: {len(lstmf_list)} entries")

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

    # Score both models on the held-out lines in every dataset font.
    results = {"name": name, "model": str(output_dir / f"{name}.traineddata"), "scores": {}}
    for index, (font_path, _) in enumerate(fonts):
        label = font_label(index, font_path)
        images = render_lines(control_lines, font_path, work / "control" / label,
                              font_size, prefix="ctrl")
        baseline = score_set(images, "eng", system_tessdata)
        trained = score_set(images, name, output_dir)
        results["scores"][label] = {"baseline": baseline, "trained": trained}
        log(f"{label}: before {baseline['exact']}/{len(images)} exact, "
            f"after {trained['exact']}/{len(images)} exact")
    return results
