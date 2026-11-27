"""Desktop window for training Tesseract on a set of fonts.

Runs on Windows. Training itself runs inside WSL, because Tesseract's training
tools are Linux programs. Drop font files anywhere in the window to add them,
set a weight for each, enter a model name, paste or generate a corpus, and
press "adjust the font".

Requirements (Windows): pip install -r requirements-gui.txt
Requirements (WSL): see README.md
"""

import json
import os
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
LOG_DIR = DATA / "logs"
FONT_EXTENSIONS = (".ttf", ".otf")
GENERATED_LINES = 200
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def to_wsl(path) -> str:
    """Convert a Windows path such as C:\\Users\\me to /mnt/c/Users/me."""
    text = str(path).replace("\\", "/")
    if not re.match(r"^[A-Za-z]:/", text):
        text = str(Path(path).resolve()).replace("\\", "/")
    if not re.match(r"^[A-Za-z]:/", text):
        raise ValueError(f"Not a Windows drive path: {path}")
    return f"/mnt/{text[0].lower()}{text[2:]}"


def build_command(args):
    """The WSL command that runs the trainer CLI inside the repo folder."""
    return ["wsl", "--cd", to_wsl(REPO), "--", "python3", "-m", "trainer.cli", *args]


def generate_lines(count=GENERATED_LINES):
    """Minecraft and mod-loader lines from tools/make_corpus.py, with a new seed each time."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.make_corpus import generate
    return generate(count, random.randrange(1, 10**9))


def clean_line(raw):
    """One output line from WSL. Some WSL output arrives as UTF-16, which leaves a NUL
    after every character; removing the NULs makes it readable."""
    return raw.replace("\x00", "").rstrip()


class TrainWorker(QThread):
    """Runs the trainer in the background so the window stays responsive."""
    line = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args

    def run(self):
        command = build_command(self.args)
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW,
            )
        except FileNotFoundError:
            self.failed.emit("WSL was not found. Install WSL and Ubuntu first (see README).")
            return

        result = None
        errors = []
        for raw in process.stdout:
            text = clean_line(raw)
            if text.startswith("RESULT:"):
                result = json.loads(text[len("RESULT:"):])
            elif text.startswith("ERROR:"):
                errors.append(text)
                self.line.emit(text)
            elif text:
                self.line.emit(text)

        code = process.wait()
        if code == 0 and result is not None:
            self.done.emit(result)
        else:
            detail = errors[-1] if errors else f"training stopped with exit code {code}"
            self.failed.emit(detail)


def font_paths(urls):
    """Local paths of the dropped URLs that are font files."""
    paths = [url.toLocalFile() for url in urls]
    return [p for p in paths if p.lower().endswith(FONT_EXTENSIONS)]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesseract Font Trainer")
        self.resize(900, 900)
        self.worker = None
        self.run_log = None
        self.run_stamp = None
        self.setAcceptDrops(True)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Font", "Weight"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(200)
        # Let drops reach the window, so any font dropped anywhere is added.
        self.table.setAcceptDrops(False)
        self.table.viewport().setAcceptDrops(False)

        add_font = QPushButton("Add font...")
        add_font.clicked.connect(self.browse_fonts)
        remove_font = QPushButton("Remove selected")
        remove_font.clicked.connect(self.remove_selected)

        self.name = QLineEdit("minecraft")

        self.corpus = QPlainTextEdit()
        self.corpus.setPlaceholderText(
            "One text line per row, at least 10. English or Latin script only.")
        self.corpus.setAcceptDrops(False)
        self.corpus.viewport().setAcceptDrops(False)
        load_corpus = QPushButton("Load text file...")
        load_corpus.clicked.connect(self.load_corpus)
        generate = QPushButton(f"Generate {GENERATED_LINES} lines")
        generate.clicked.connect(self.generate_corpus)

        self.adjust = QPushButton("adjust the font")
        self.adjust.setMinimumHeight(56)
        self.adjust.clicked.connect(self.start_training)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here.")
        open_logs = QPushButton("Open logs folder")
        open_logs.clicked.connect(self.open_logs)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Drop font files anywhere in this window. Weight = how many times each "
            "training line is rendered in that font."))
        layout.addWidget(self.table)
        font_buttons = QHBoxLayout()
        font_buttons.addWidget(add_font)
        font_buttons.addWidget(remove_font)
        font_buttons.addStretch()
        layout.addLayout(font_buttons)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model name:"))
        model_row.addWidget(self.name)
        layout.addLayout(model_row)

        layout.addWidget(QLabel("Corpus (shared by all fonts):"))
        layout.addWidget(self.corpus)
        corpus_buttons = QHBoxLayout()
        corpus_buttons.addWidget(load_corpus)
        corpus_buttons.addWidget(generate)
        corpus_buttons.addStretch()
        layout.addLayout(corpus_buttons)

        layout.addWidget(self.adjust)
        layout.addWidget(self.status)
        layout.addWidget(open_logs)
        layout.addWidget(self.log)

    def dragEnterEvent(self, event):
        if font_paths(event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = font_paths(event.mimeData().urls())
        if paths:
            self.add_fonts(paths)
            event.acceptProposedAction()

    def add_fonts(self, paths):
        existing = {self.table.item(r, 0).data(Qt.UserRole)
                    for r in range(self.table.rowCount())}
        for path in paths:
            if path not in existing:
                self._append_row(path, 1)
                existing.add(path)

    def browse_fonts(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose fonts", "", "Fonts (*.ttf *.otf)")
        if paths:
            self.add_fonts(paths)

    def _append_row(self, path, weight):
        row = self.table.rowCount()
        self.table.insertRow(row)
        item = QTableWidgetItem(Path(path).name)
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.table.setItem(row, 0, item)
        spin = QSpinBox()
        spin.setRange(1, 10)
        spin.setValue(weight)
        self.table.setCellWidget(row, 1, spin)

    def remove_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def dataset(self):
        """List of (path, weight) for every row in the table."""
        fonts = []
        for row in range(self.table.rowCount()):
            path = self.table.item(row, 0).data(Qt.UserRole)
            weight = self.table.cellWidget(row, 1).value()
            fonts.append((path, weight))
        return fonts

    def load_corpus(self):
        path, _ = QFileDialog.getOpenFileName(self, "Corpus", "", "Text (*.txt)")
        if path:
            self.corpus.setPlainText(Path(path).read_text(encoding="utf-8"))

    def generate_corpus(self):
        new_lines = generate_lines()
        existing = self.corpus.toPlainText().rstrip("\n")
        text = "\n".join(new_lines)
        self.corpus.setPlainText(f"{existing}\n{text}" if existing else text)

    def start_training(self):
        problem = self.validate()
        if problem:
            QMessageBox.warning(self, "Cannot start", problem)
            return

        name = self.name.text().strip()
        corpus_file = DATA / f"corpus_{name}.txt"
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        corpus_file.write_text(self.corpus.toPlainText(), encoding="utf-8")

        fonts = [{"path": to_wsl(path), "weight": weight} for path, weight in self.dataset()]
        args = [
            "--fonts-json", json.dumps(fonts),
            "--name", name,
            "--corpus-file", to_wsl(corpus_file),
            "--data-root", to_wsl(DATA / "runs"),
        ]

        self.log.clear()
        self.open_run_log(name)
        self.record(f"Model: {name}")
        self.record(f"Corpus: {len([l for l in self.corpus.toPlainText().splitlines() if l.strip()])} lines")
        for path, weight in self.dataset():
            self.record(f"Font: {path} (weight {weight})")
        self.status.setText("Training... this can take a long time. Keep the window open.")
        self.adjust.setEnabled(False)
        self.worker = TrainWorker(args)
        self.worker.line.connect(self.record)
        self.worker.done.connect(self.show_results)
        self.worker.failed.connect(self.show_failure)
        self.worker.start()

    def validate(self):
        if self.table.rowCount() == 0:
            return "Add at least one font to the dataset."
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", self.name.text().strip()):
            return "Model name: use letters, digits, - or _ (max 40 characters)."
        lines = [l for l in self.corpus.toPlainText().splitlines() if l.strip()]
        if len(lines) < 10:
            return f"The corpus needs at least 10 lines. It has {len(lines)}."
        return None

    def open_run_log(self, name):
        """Every run gets a text log and a JSON result in data/logs, so you can read them later."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.run_stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        self.run_log = open(LOG_DIR / f"{self.run_stamp}.log", "w", encoding="utf-8")
        self.record(f"Run started: {self.run_stamp}")

    def record(self, text):
        self.log.appendPlainText(text)
        if self.run_log:
            self.run_log.write(text + "\n")
            self.run_log.flush()

    def close_run_log(self):
        if self.run_log:
            self.record(f"Log saved to: {LOG_DIR / (self.run_stamp + '.log')}")
            self.run_log.close()
            self.run_log = None

    def open_logs(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(LOG_DIR)
        else:
            self.status.setText(f"Logs are in {LOG_DIR}")

    def show_results(self, result):
        self.adjust.setEnabled(True)
        (LOG_DIR / f"{self.run_stamp}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        lines = ["", "Results (exact matches on held-out lines):"]
        for label, scores in result["scores"].items():
            b, t = scores["baseline"], scores["trained"]
            font = label.split("_", 1)[-1]
            lines.append(
                f"  {font}: before {b['exact']}/{b['total']} (CER {b['cer']}), "
                f"after {t['exact']}/{t['total']} (CER {t['cer']})")
        model = DATA / "runs" / result["name"] / "model" / f"{result['name']}.traineddata"
        lines.append(f"Model saved to: {model}")
        self.record("\n".join(lines))
        self.close_run_log()
        self.status.setText("Done. The log is in the logs folder.")

    def show_failure(self, detail):
        self.adjust.setEnabled(True)
        self.status.setText("Training failed. See the log below.")
        self.record(f"\nFAILED: {detail}")
        self.close_run_log()


def main():
    app = QApplication(sys.argv)
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
