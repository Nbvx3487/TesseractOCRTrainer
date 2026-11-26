"""Desktop window for training Tesseract on a set of fonts.

Runs on Windows. Training itself runs inside WSL, because Tesseract's training
tools are Linux programs. Drop fonts onto the window, set a weight for each,
paste a corpus, and press "Make good".

Requirements (Windows): pip install -r requirements-gui.txt
Requirements (WSL): see README.md
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
FONT_EXTENSIONS = (".ttf", ".otf")
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
            text = raw.rstrip()
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


class DropArea(QLabel):
    """Accepts dropped font files, or a click to browse for them."""

    def __init__(self, on_fonts, parent=None):
        super().__init__(parent)
        self.on_fonts = on_fonts
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(110)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("Drop font files here\n(or click to browse)")
        self.setStyleSheet("border: 2px dashed #888; border-radius: 8px; padding: 10px;")

    def dragEnterEvent(self, event):
        if self._font_paths(event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = self._font_paths(event.mimeData().urls())
        if paths:
            self.on_fonts(paths)

    def mousePressEvent(self, event):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose fonts", "", "Fonts (*.ttf *.otf)")
        if paths:
            self.on_fonts(paths)

    @staticmethod
    def _font_paths(urls):
        paths = [url.toLocalFile() for url in urls]
        return [p for p in paths if p.lower().endswith(FONT_EXTENSIONS)]


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesseract Font Trainer")
        self.resize(900, 900)
        self.worker = None

        self.drop = DropArea(self.add_fonts)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Font", "Weight"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(160)

        add_font = QPushButton("Add font...")
        add_font.clicked.connect(self.browse_fonts)
        remove_font = QPushButton("Remove selected")
        remove_font.clicked.connect(self.remove_selected)

        self.name = QLineEdit("minecraft")

        self.corpus = QPlainTextEdit()
        self.corpus.setPlaceholderText(
            "One text line per row, at least 10. English or Latin script only.")
        load_corpus = QPushButton("Load text file...")
        load_corpus.clicked.connect(self.load_corpus)

        self.make_good = QPushButton("Make good")
        self.make_good.setMinimumHeight(56)
        self.make_good.clicked.connect(self.start_training)

        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here.")

        layout = QVBoxLayout(self)
        layout.addWidget(self.drop)

        layout.addWidget(QLabel("Fonts in the dataset. Weight = how many times each line is trained in that font."))
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

        layout.addWidget(QLabel("Corpus:"))
        layout.addWidget(self.corpus)
        layout.addWidget(load_corpus)

        layout.addWidget(self.make_good)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

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
        self.status.setText("Training... this can take a long time. Keep the window open.")
        self.make_good.setEnabled(False)
        self.worker = TrainWorker(args)
        self.worker.line.connect(self.log.appendPlainText)
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

    def show_results(self, result):
        self.make_good.setEnabled(True)
        lines = ["", "Results (exact matches on held-out lines):"]
        for label, scores in result["scores"].items():
            b, t = scores["baseline"], scores["trained"]
            font = label.split("_", 1)[-1]
            lines.append(
                f"  {font}: before {b['exact']}/{b['total']} (CER {b['cer']}), "
                f"after {t['exact']}/{t['total']} (CER {t['cer']})")
        model = DATA / "runs" / result["name"] / "model" / f"{result['name']}.traineddata"
        lines.append(f"Model saved to: {model}")
        self.log.appendPlainText("\n".join(lines))
        self.status.setText("Done.")

    def show_failure(self, detail):
        self.make_good.setEnabled(True)
        self.status.setText("Training failed. See the log below.")
        self.log.appendPlainText(f"\nFAILED: {detail}")


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
