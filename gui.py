"""Desktop window for training Tesseract on a set of fonts.

Runs on Windows. Training itself runs inside WSL, because Tesseract's training
tools are Linux programs. Common Windows fonts are added at startup. Drop more
font files anywhere in the window, set a weight for each, enter a model name,
paste or generate a corpus, and press "adjust the font".

Double-click gui.pyw to open it without a console window.

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

try:
    import winreg
except ImportError:  # not on Windows, so the theme cannot be read
    winreg = None

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLayout, QLineEdit, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
LOG_DIR = DATA / "logs"
FONT_EXTENSIONS = (".ttf", ".otf")
GENERATED_LINES = 200
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOWS_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
COMMON_FONTS = [
    "arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf",
    "tahoma.ttf", "calibri.ttf", "consola.ttf", "segoeui.ttf",
]

# Colours follow the Windows settings look: light or dark, with a blue accent.
LIGHT = dict(bg="#F3F3F3", card="#FFFFFF", border="#E5E5E5", hover="#F5F5F5",
             text="#1A1A1A", subtext="#5C5C5C", accent="#0067C0", on_accent="#FFFFFF")
DARK = dict(bg="#202020", card="#2B2B2B", border="#3B3B3B", hover="#353535",
            text="#FFFFFF", subtext="#C5C5C5", accent="#4CC2FF", on_accent="#000000")


def system_is_dark():
    """True when Windows is set to dark mode for apps."""
    if winreg is None:
        return False
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            return winreg.QueryValueEx(key, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def stylesheet(t):
    return f"""
QWidget {{ background: {t['bg']}; color: {t['text']}; font-family: "Segoe UI"; }}
QLabel {{ background: transparent; }}
QListWidget#nav {{ background: transparent; border: none; outline: 0; padding: 8px 4px; }}
QListWidget#nav::item {{ padding: 8px 12px; border-radius: 4px; margin: 2px 0; }}
QListWidget#nav::item:selected {{ background: {t['card']}; color: {t['accent']}; }}
QListWidget#nav::item:hover {{ background: {t['hover']}; }}
QLabel#pageTitle {{ font-size: 24px; font-weight: 600; }}
QLabel#pageDesc, QLabel#cardDesc {{ color: {t['subtext']}; }}
QLabel#cardTitle {{ font-weight: 600; }}
QFrame#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 8px; }}
QPushButton {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 6px 14px; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:disabled {{ color: {t['subtext']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']}; border: none; font-weight: 600; padding: 12px; }}
QPushButton#primary:disabled {{ background: {t['border']}; }}
QLineEdit, QPlainTextEdit, QTableWidget, QSpinBox {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 4px; }}
QTableWidget {{ gridline-color: {t['border']}; }}
QHeaderView::section {{ background: {t['card']}; color: {t['subtext']}; border: none; border-bottom: 1px solid {t['border']}; padding: 6px; font-weight: 600; }}
"""


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
        self.resize(1000, 720)
        self.worker = None
        self.run_log = None
        self.run_stamp = None
        self.setAcceptDrops(True)

        # Fonts page
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Font", "Weight", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(260)
        # Let drops reach the window, so any font dropped anywhere is added.
        self.table.setAcceptDrops(False)
        self.table.viewport().setAcceptDrops(False)
        self.add_font_button = QPushButton("Add font...")
        self.add_font_button.clicked.connect(self.browse_fonts)

        # Corpus page
        self.corpus = QPlainTextEdit()
        self.corpus.setPlaceholderText(
            "One text line per row, at least 10. English or Latin script only.")
        self.corpus.setAcceptDrops(False)
        self.corpus.viewport().setAcceptDrops(False)
        self.corpus.setMinimumHeight(360)
        self.load_corpus_button = QPushButton("Load text file...")
        self.load_corpus_button.clicked.connect(self.load_corpus)
        self.generate_button = QPushButton(f"Generate {GENERATED_LINES} lines")
        self.generate_button.clicked.connect(self.generate_corpus)

        # Training page
        self.name = QLineEdit("minecraft")
        self.name.setFixedWidth(240)
        self.adjust = QPushButton("adjust the font")
        self.adjust.setObjectName("primary")
        self.adjust.clicked.connect(self.start_training)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here.")
        self.log.setMinimumHeight(260)
        self.open_logs_button = QPushButton("Open logs folder")
        self.open_logs_button.clicked.connect(self.open_logs)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(170)
        self.pages = QStackedWidget()
        for title, page in [("Fonts", self._fonts_page()),
                            ("Corpus", self._corpus_page()),
                            ("Training", self._training_page())]:
            self.nav.addItem(title)
            self.pages.addWidget(page)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.nav)
        root.addWidget(self.pages, 1)

        self.add_common_fonts()

    # ----- page layout helpers -----

    def _page(self, title, subtitle):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        head = QLabel(title)
        head.setObjectName("pageTitle")
        sub = QLabel(subtitle)
        sub.setObjectName("pageDesc")
        sub.setWordWrap(True)
        layout.addWidget(head)
        layout.addWidget(sub)
        return widget, layout

    def _card(self, title, description, body):
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        head = QLabel(title)
        head.setObjectName("cardTitle")
        layout.addWidget(head)
        if description:
            desc = QLabel(description)
            desc.setObjectName("cardDesc")
            desc.setWordWrap(True)
            layout.addWidget(desc)
        if isinstance(body, QLayout):
            layout.addLayout(body)
        else:
            layout.addWidget(body)
        return frame

    def _setting(self, title, description, control):
        """A settings row as in Windows: text on the left, the control on the right."""
        row = QHBoxLayout()
        text = QVBoxLayout()
        text.setSpacing(2)
        head = QLabel(title)
        head.setObjectName("cardTitle")
        text.addWidget(head)
        if description:
            desc = QLabel(description)
            desc.setObjectName("cardDesc")
            desc.setWordWrap(True)
            text.addWidget(desc)
        row.addLayout(text, 1)
        row.addWidget(control, 0, Qt.AlignVCenter)
        return row

    def _fonts_page(self):
        page, layout = self._page(
            "Fonts",
            "Drop font files anywhere in this window to add them.")
        body = QVBoxLayout()
        body.addWidget(self.table)
        buttons = QHBoxLayout()
        buttons.addWidget(self.add_font_button)
        buttons.addStretch()
        body.addLayout(buttons)
        layout.addWidget(self._card(
            "Fonts in the dataset",
            "Weight = how many times each training line is rendered in that font. "
            "Common Windows fonts are added at startup.",
            body))
        layout.addStretch()
        return page

    def _corpus_page(self):
        page, layout = self._page(
            "Corpus",
            "One text line per row. The same corpus is used for every font.")
        body = QVBoxLayout()
        body.addWidget(self.corpus)
        buttons = QHBoxLayout()
        buttons.addWidget(self.load_corpus_button)
        buttons.addWidget(self.generate_button)
        buttons.addStretch()
        body.addLayout(buttons)
        layout.addWidget(self._card("Shared corpus", None, body))
        return page

    def _training_page(self):
        page, layout = self._page(
            "Training",
            "Trains every font in the list, then scores before and after on held-out lines.")
        layout.addWidget(self._card(
            "Model",
            None,
            self._setting("Model name",
                          "Letters, digits, - or _. One name for the whole run.",
                          self.name)))

        run_body = QVBoxLayout()
        run_body.addWidget(self.adjust)
        run_body.addWidget(self.status)
        layout.addWidget(self._card(
            "Run",
            "Training can take a long time. Keep the window open while it runs.",
            run_body))

        log_body = QVBoxLayout()
        log_body.addWidget(self.log)
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_logs_button)
        buttons.addStretch()
        log_body.addLayout(buttons)
        layout.addWidget(self._card(
            "Log",
            "Each run is also saved to the logs folder.",
            log_body))
        return page

    # ----- fonts -----

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

    def add_common_fonts(self):
        """Add the common Windows fonts that are installed, with weight 1."""
        found = [str(WINDOWS_FONTS / name) for name in COMMON_FONTS
                 if (WINDOWS_FONTS / name).exists()]
        self.add_fonts(found)

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
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda checked=False, p=path: self.remove_path(p))
        self.table.setCellWidget(row, 2, remove)

    def remove_path(self, path):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.UserRole) == path:
                self.table.removeRow(row)
                return

    def dataset(self):
        """List of (path, weight) for every row in the table."""
        fonts = []
        for row in range(self.table.rowCount()):
            path = self.table.item(row, 0).data(Qt.UserRole)
            weight = self.table.cellWidget(row, 1).value()
            fonts.append((path, weight))
        return fonts

    # ----- corpus -----

    def load_corpus(self):
        path, _ = QFileDialog.getOpenFileName(self, "Corpus", "", "Text (*.txt)")
        if path:
            self.corpus.setPlainText(Path(path).read_text(encoding="utf-8"))

    def generate_corpus(self):
        new_lines = generate_lines()
        existing = self.corpus.toPlainText().rstrip("\n")
        text = "\n".join(new_lines)
        self.corpus.setPlainText(f"{existing}\n{text}" if existing else text)

    # ----- training -----

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
        corpus_lines = [l for l in self.corpus.toPlainText().splitlines() if l.strip()]
        self.record(f"Corpus: {len(corpus_lines)} lines")
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
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)
    app.setStyleSheet(stylesheet(DARK if system_is_dark() else LIGHT))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
