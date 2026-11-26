"""Desktop window for training Tesseract on a font.

Runs on Windows. Training itself runs inside WSL, because Tesseract's training
tools are Linux programs. Drop a font onto the window, adjust settings, and
press "Make good".

Requirements (Windows): pip install -r requirements-gui.txt
Requirements (WSL): see README.md
"""

import json
import re
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
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
    """Accepts a dropped font file, or a click to browse for one."""

    def __init__(self, on_font, parent=None):
        super().__init__(parent)
        self.on_font = on_font
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(90)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Drop a .ttf or .otf file here, or click to browse")
        self.set_text("Drop a font file here\n(or click to browse)")

    def set_text(self, text):
        self.setText(text)
        self.setStyleSheet("border: 2px dashed #888; border-radius: 8px; padding: 8px;")

    def dragEnterEvent(self, event):
        if self._first_font(event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        path = self._first_font(event.mimeData().urls())
        if path:
            self.on_font(path)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a font", "", "Fonts (*.ttf *.otf)")
        if path:
            self.on_font(path)

    @staticmethod
    def _first_font(urls):
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith(FONT_EXTENSIONS):
                return path
        return None


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesseract Font Trainer")
        self.resize(760, 760)
        self.font_path = None
        self.worker = None

        self.drop = DropArea(self.set_font)
        self.font_label = QLabel("No font chosen yet.")

        self.name = QLineEdit("minecraft")
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 200)
        self.font_size.setValue(32)
        self.iterations = QSpinBox()
        self.iterations.setRange(50, 20000)
        self.iterations.setValue(400)

        self.control_list = QListWidget()
        self.control_list.setMaximumHeight(70)
        add_control = QPushButton("Add check font...")
        add_control.clicked.connect(self.add_control_font)
        remove_control = QPushButton("Remove")
        remove_control.clicked.connect(self.remove_control_font)

        self.corpus = QPlainTextEdit()
        self.corpus.setPlaceholderText(
            "One text line per row, at least 10. English or Latin script only.")
        load_corpus = QPushButton("Load text file...")
        load_corpus.clicked.connect(self.load_corpus)

        self.make_good = QPushButton("Make good")
        self.make_good.setMinimumHeight(40)
        self.make_good.clicked.connect(self.start_training)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here.")

        self.status = QLabel("")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.drop)
        layout.addWidget(self.font_label)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("Model name:"))
        settings.addWidget(self.name)
        settings.addWidget(QLabel("Font size (px):"))
        settings.addWidget(self.font_size)
        settings.addWidget(QLabel("Iterations:"))
        settings.addWidget(self.iterations)
        layout.addLayout(settings)

        layout.addWidget(QLabel("Check fonts (also tested, to catch forgetting):"))
        layout.addWidget(self.control_list)
        buttons = QHBoxLayout()
        buttons.addWidget(add_control)
        buttons.addWidget(remove_control)
        buttons.addStretch()
        layout.addLayout(buttons)

        layout.addWidget(QLabel("Corpus:"))
        layout.addWidget(self.corpus)
        layout.addWidget(load_corpus)

        layout.addWidget(self.make_good)
        layout.addWidget(self.status)
        layout.addWidget(self.log)

    def set_font(self, path):
        self.font_path = path
        self.font_label.setText(f"Font: {Path(path).name}")
        self.drop.set_text(f"Loaded: {Path(path).name}\n(drop another to replace)")

    def add_control_font(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Check fonts", "", "Fonts (*.ttf *.otf)")
        existing = {self.control_list.item(i).text() for i in range(self.control_list.count())}
        for path in paths:
            if path not in existing:
                self.control_list.addItem(path)

    def remove_control_font(self):
        for item in self.control_list.selectedItems():
            self.control_list.takeItem(self.control_list.row(item))

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
        run_dir = DATA / "runs"
        corpus_file = DATA / f"corpus_{name}.txt"
        corpus_file.parent.mkdir(parents=True, exist_ok=True)
        corpus_file.write_text(self.corpus.toPlainText(), encoding="utf-8")

        args = [
            "--font", to_wsl(self.font_path),
            "--name", name,
            "--corpus-file", to_wsl(corpus_file),
            "--data-root", to_wsl(run_dir),
            "--font-size", str(self.font_size.value()),
            "--iterations", str(self.iterations.value()),
        ]
        for i in range(self.control_list.count()):
            args += ["--control-font", to_wsl(self.control_list.item(i).text())]

        self.log.clear()
        self.status.setText("Training... this can take a long time. Keep the window open.")
        self.make_good.setEnabled(False)
        self.worker = TrainWorker(args)
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(self.show_results)
        self.worker.failed.connect(self.show_failure)
        self.worker.start()

    def validate(self):
        if not self.font_path:
            return "Drop a font file first."
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
            lines.append(
                f"  {label}: before {b['exact']}/{b['total']} (CER {b['cer']}), "
                f"after {t['exact']}/{t['total']} (CER {t['cer']})")
        model_win = DATA / "runs" / result["name"] / "model" / f"{result['name']}.traineddata"
        lines.append(f"Model saved to: {model_win}")
        self.log.appendPlainText("\n".join(lines))
        self.status.setText("Done.")

    def show_failure(self, detail):
        self.make_good.setEnabled(True)
        self.status.setText("Training failed. See the log below.")
        self.log.appendPlainText(f"\nFAILED: {detail}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
