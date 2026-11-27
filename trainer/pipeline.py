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
import time
from pathlib import Path

from .corpus import split_corpus
from .render import font_cache_dir, render_lines
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


def score_and_log(label, images, lang, tessdata_dir, when, log):
    """Score the held-out lines and log each line as PASS or FAIL."""
    def on_line(checked, total):
        log(f"  checked {checked}/{total} lines in {label} ({when})")

    scores = score_set(images, lang, tessdata_dir, on_line=on_line)
    log(f"{label} {when}: {scores['exact']}/{scores['total']} lines pass, CER {scores['cer']}")
    for line in scores["lines"]:
        if line["pass"]:
            log(f"  PASS  {line['expected']}")
        else:
            log(f"  FAIL  expected: {line['expected']}")
            log(f"        got:      {line['got']}")
    return scores


def font_label(index, font_path):
    return f"{index}_{Path(font_path).stem}"


def train_font(name, fonts, corpus, data_root, log, font_size=32, iterations=400, check_every=0,
               cache_root=None):
    """Run the full pipeline.

    fonts: list of (font_path, weight) pairs. Weight is an integer, 1 or more.
    check_every: if above 0, training stops every this many iterations, the checkpoint is
    scored on the held-out lines, and training resumes from it (series mode).
    Returns a dict with per-font scores, the checks, and the path to the trained model.
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

    # Progress units: one per held-out line checked before training, one per training
    # line prepared, one per training iteration, and one per held-out line checked after.
    font_count = len(fonts)
    prepared_units = font_count * len(train_lines)
    checks_total = -(-iterations // check_every) if check_every > 0 else 0  # ceiling
    # Every step counts: render and score the control lines (before and after training),
    # render and prepare the training lines, train, and each checkpoint check.
    total = (font_count * len(control_lines) * 3 + prepared_units * 2 + iterations
             + checks_total * font_count * len(control_lines))
    done = 0
    # Time spent on each step, saved with the result so the next estimate can use it.
    timings = {"render_seconds": 0.0, "rendered_images": 0, "prepare_seconds": 0.0,
               "prepared_images": 0, "train_seconds": 0.0, "iterations": 0,
               "score_seconds": 0.0, "scored_images": 0}

    def cache_dir_for(font_path):
        if cache_root is None:
            return None
        return font_cache_dir(cache_root, font_path, font_size)

    # Held-out check before training: the untrained English model, font by font.
    control_images = {}
    baseline_scores = {}
    for index, (font_path, _) in enumerate(fonts):
        label = font_label(index, font_path)
        log(f"Rendering {len(control_lines)} held-out lines in {label}")
        render_start = time.monotonic()
        images = render_lines(control_lines, font_path, work / "control" / label,
                              font_size, prefix="ctrl", log=log, label=label,
                              progress=lambda cur, cnt: report(log, "render", label, cur, cnt,
                                                               done + cur, total),
                              cache_dir=cache_dir_for(font_path))
        timings["render_seconds"] += time.monotonic() - render_start
        timings["rendered_images"] += len(images)
        log(f"Rendered {len(images)} held-out images in {label}")
        control_images[label] = images
        done += len(images)
        report(log, "score", label, 0, len(images), done, total)
        score_start = time.monotonic()
        baseline_scores[label] = score_and_log(label, images, "eng", system_tessdata,
                                               "before training", log)
        timings["score_seconds"] += time.monotonic() - score_start
        timings["scored_images"] += len(images)
        done += len(images)
        report(log, "score", label, len(images), len(images), done, total)

    # Render training images per font, and list each one weight times.
    lstmf_list = []
    for index, (font_path, weight) in enumerate(fonts):
        label = font_label(index, font_path)
        train_dir = work / "train" / label
        log(f"Rendering {len(train_lines)} training lines in {label} (weight {weight})")
        render_start = time.monotonic()
        tifs = render_lines(train_lines, font_path, train_dir, font_size, prefix="train",
                            log=log, label=label,
                            progress=lambda cur, cnt: report(log, "render", label, cur, cnt,
                                                             done + cur, total),
                            cache_dir=cache_dir_for(font_path))
        done += len(tifs)
        timings["render_seconds"] += time.monotonic() - render_start
        timings["rendered_images"] += len(tifs)
        prepare_start = time.monotonic()
        log(f"Rendered {len(tifs)} training images in {label}")
        for number, tif in enumerate(tifs, start=1):
            run(["tesseract", str(tif), str(tif.with_suffix("")),
                 "--psm", "13", "lstm.train"], log, quiet=True)
            entry = str(tif.with_suffix(".lstmf"))
            lstmf_list.extend([entry] * weight)
            done += 1
            report(log, "prepare", label, number, len(tifs), done, total)
            log(f"  prepared {number}/{len(tifs)} lines in {label}: {tif.name}")
        timings["prepare_seconds"] += time.monotonic() - prepare_start
        timings["prepared_images"] += len(tifs)
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
                   done + current + state["checks_done"], total)

    report(log, "train", "all fonts", 0, iterations, done, total)
    state = {"last": 0, "finished": False, "checks_done": 0}

    def on_chunk_line(line):
        match = ITERATION_LINE.search(line)
        if match:
            state["last"] = int(match.group(1))
        if "Finished!" in line:
            state["finished"] = True
        on_training_line(line)

    def run_check(iteration):
        """Turn the checkpoint into a model and score it on the held-out lines."""
        check_start = time.monotonic()
        check_dir = work / "checks" / f"iteration_{iteration:04d}"
        check_dir.mkdir(parents=True, exist_ok=True)
        run(["lstmtraining", "--stop_training",
             "--continue_from", str(checkpoints / f"{name}_checkpoint"),
             "--traineddata", str(base_dir / "eng.traineddata"),
             "--model_output", str(check_dir / f"{name}.traineddata")], log, quiet=True)
        entry = {"iteration": iteration, "fonts": {}}
        for index, (font_path, _) in enumerate(fonts):
            label = font_label(index, font_path)
            score = score_set(control_images[label], name, check_dir)
            entry["fonts"][label] = {"exact": score["exact"], "total": score["total"],
                                     "cer": score["cer"]}
            log(f"Check at iteration {iteration}: {label} {score['exact']}/{score['total']} "
                f"lines pass, CER {score['cer']}")
        timings["score_seconds"] += time.monotonic() - check_start
        timings["scored_images"] += sum(len(images) for images in control_images.values())
        return entry

    checks = []
    if check_every <= 0:
        train_start = time.monotonic()
        run(["lstmtraining",
             "--model_output", str(checkpoints / name),
             "--continue_from", str(base_dir / "eng.lstm"),
             "--traineddata", str(base_dir / "eng.traineddata"),
             "--train_listfile", str(list_file),
             "--max_iterations", str(iterations)], log, on_line=on_training_line)
        timings["train_seconds"] += time.monotonic() - train_start
    else:
        source = base_dir / "eng.lstm"
        target = 0
        while target < iterations and not state["finished"]:
            target = min(target + check_every, iterations)
            log(f"Training to iteration {target} of {iterations}")
            chunk_start = time.monotonic()
            run(["lstmtraining",
                 "--model_output", str(checkpoints / name),
                 "--continue_from", str(source),
                 "--traineddata", str(base_dir / "eng.traineddata"),
                 "--train_listfile", str(list_file),
                 "--max_iterations", str(target)], log, on_line=on_chunk_line)
            timings["train_seconds"] += time.monotonic() - chunk_start
            source = checkpoints / f"{name}_checkpoint"
            checks.append(run_check(state["last"] or target))
            state["checks_done"] += font_count * len(control_lines)
            (work / "metrics.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    done += iterations + state["checks_done"]

    output_dir = work / "model"
    output_dir.mkdir()
    run(["lstmtraining", "--stop_training",
         "--continue_from", str(checkpoints / f"{name}_checkpoint"),
         "--traineddata", str(base_dir / "eng.traineddata"),
         "--model_output", str(output_dir / f"{name}.traineddata")], log)

    # Held-out check after training, with the same lines as before.
    results = {"name": name, "model": str(output_dir / f"{name}.traineddata"), "scores": {},
               "checks": checks,
               "timings": dict(timings, iterations=iterations, fonts=font_count,
                               lines=len(train_lines) + len(control_lines),
                               train_lines=len(train_lines), control_lines=len(control_lines),
                               check_every=check_every, checks=len(checks))}
    for index, (font_path, _) in enumerate(fonts):
        label = font_label(index, font_path)
        images = control_images[label]
        report(log, "score", label, 0, len(images), done, total)
        score_start = time.monotonic()
        trained = score_and_log(label, images, name, output_dir, "after training", log)
        timings["score_seconds"] += time.monotonic() - score_start
        timings["scored_images"] += len(images)
        done += len(images)
        report(log, "score", label, len(images), len(images), done, total)
        results["scores"][label] = {"baseline": baseline_scores[label], "trained": trained}
    return results
