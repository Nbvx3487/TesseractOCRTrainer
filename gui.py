"""Desktop window for training Tesseract on a set of fonts.

Runs on Windows. Training itself runs inside WSL, because Tesseract's training
tools are Linux programs.

Layout: resource limits on the left. On the right, from top to bottom: the main
action, the fonts to train, the corpus, the training settings, the results of
the last run, and the log. Common Windows fonts are added at startup, and any
font file dropped on the window is added to the list.

Only one copy of the window runs at a time. Double-click gui.pyw to open it
without a console window.

Requirements (Windows): pip install -r requirements-gui.txt
Requirements (WSL): see README.md
"""

# Draw with software rendering rather than the graphics driver. This avoided a
# crash on the Windows PC. It must be set before PySide6 loads.
import os
os.environ.setdefault("QT_OPENGL", "software")

import json
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
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLayout, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
LOG_DIR = DATA / "logs"
FONT_EXTENSIONS = (".ttf", ".otf")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOWS_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
COMMON_FONTS = [
    "arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf",
    "tahoma.ttf", "calibri.ttf", "consola.ttf", "segoeui.ttf",
]
SINGLE_INSTANCE_NAME = "TesseractFontTrainer"
DEFAULT_ITERATIONS = 400
DEFAULT_FONT_SIZE = 32
DEFAULT_GENERATED_LINES = 200
DEFAULT_MEMORY_GB = 16
DEFAULT_THREADS = 4

# Colours follow the Windows settings look: light or dark, with a blue accent.
LIGHT = dict(bg="#F3F3F3", card="#FFFFFF", border="#E5E5E5", hover="#F5F5F5",
             text="#1A1A1A", subtext="#5C5C5C", accent="#0067C0", on_accent="#FFFFFF",
             banner="#E8F1FB")
DARK = dict(bg="#202020", card="#2B2B2B", border="#3B3B3B", hover="#353535",
            text="#FFFFFF", subtext="#C5C5C5", accent="#4CC2FF", on_accent="#000000",
            banner="#253A4E")


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
QScrollArea {{ background: transparent; border: none; }}
QFrame#panel {{ background: {t['card']}; border-right: 1px solid {t['border']}; }}
QLabel#panelTitle {{ font-size: 18px; font-weight: 600; }}
QLabel#pageTitle {{ font-size: 24px; font-weight: 600; }}
QLabel#pageDesc, QLabel#cardDesc {{ color: {t['subtext']}; }}
QLabel#cardTitle {{ font-weight: 600; }}
QFrame#banner {{ background: {t['banner']}; border-radius: 8px; }}
QFrame#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 8px; }}
QPushButton {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 4px; padding: 6px 14px; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:disabled {{ color: {t['subtext']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']}; border: none; font-weight: 600; padding: 12px 24px; }}
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


def build_command(args, threads):
    """The WSL command that runs the trainer CLI inside the repo folder, with the CPU
    thread limit set through OMP_THREAD_LIMIT (read by Tesseract's OpenMP code)."""
    return ["wsl", "--cd", to_wsl(REPO), "--", "env", f"OMP_THREAD_LIMIT={threads}",
            "python3", "-m", "trainer.cli", *args]


def wslconfig_text(text, memory_gb, processors):
    """Return the .wslconfig text with memory and processors set under [wsl2].
    Other settings in the file are kept."""
    wanted = [f"memory={memory_gb}GB", f"processors={processors}"]
    lines = text.splitlines()
    if not any(line.strip().lower() == "[wsl2]" for line in lines):
        lines.append("[wsl2]")
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "[wsl2]")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("[")),
               len(lines))
    body = [line for line in lines[start + 1:end]
            if line.split("=", 1)[0].strip().lower() not in ("memory", "processors")]
    lines = lines[:start + 1] + wanted + body + lines[end:]
    return "\n".join(lines).rstrip() + "\n"


def generate_lines(count):
    """Minecraft and mod-loader lines from tools/make_corpus.py, with a new seed each time."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.make_corpus import generate
    return generate(count, random.randrange(1, 10**9))


def clean_line(raw):
    """One output line from WSL. Some WSL output arrives as UTF-16, which leaves a NUL
    after every character; removing the NULs makes it readable."""
    return raw.replace("\x00", "").rstrip()


def font_paths(urls):
    """Local paths of the dropped URLs that are font files."""
    paths = [url.toLocalFile() for url in urls]
    return [p for p in paths if p.lower().endswith(FONT_EXTENSIONS)]


def another_copy_is_open():
    """True when another copy of this window is already running."""
    socket = QLocalSocket()
    socket.connectToServer(SINGLE_INSTANCE_NAME)
    connected = socket.waitForConnected(500)
    socket.abort()
    return connected


class TrainWorker(QThread):
    """Runs the trainer in the background so the window stays responsive."""
    line = Signal(str)
    progress = Signal(dict)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, args, threads, parent=None):
        super().__init__(parent)
        self.args = args
        self.threads = threads
        self.process = None
        self.stopped = False

    def stop(self):
        """Stop the run. The trainer and the programs it started run inside WSL, so they
        are killed there. Killing only the Windows side leaves them running and keeps the
        output pipe open."""
        self.stopped = True
        subprocess.Popen(
            ["wsl", "--", "pkill", "-f", "trainer.cli|lstmtraining|text2image|tesseract"],
            creationflags=CREATE_NO_WINDOW)
        if self.process is not None:
            self.process.kill()

    def run(self):
        command = build_command(self.args, self.threads)
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW,
            )
        except FileNotFoundError:
            self.failed.emit("WSL was not found. Install WSL and Ubuntu first (see README).")
            return

        result = None
        errors = []
        for raw in self.process.stdout:
            text = clean_line(raw)
            if text.startswith("PROGRESS:"):
                self.progress.emit(json.loads(text[len("PROGRESS:"):]))
            elif text.startswith("RESULT:"):
                result = json.loads(text[len("RESULT:"):])
            elif text.startswith("ERROR:"):
                errors.append(text)
                self.line.emit(text)
            elif text:
                self.line.emit(text)

        code = self.process.wait()
        if self.stopped:
            self.failed.emit("Training stopped.")
        elif code == 0 and result is not None:
            self.done.emit(result)
        else:
            detail = errors[-1] if errors else f"training stopped with exit code {code}"
            self.failed.emit(detail)


def font_paths(urls):
    """Local paths of the dropped URLs that are font files."""
    paths = [url.toLocalFile() for url in urls]
    return [p for p in paths if p.lower().endswith(FONT_EXTENSIONS)]


def another_copy_is_open():
    """True when another copy of this window is already running."""
    socket = QLocalSocket()
    socket.connectToServer(SINGLE_INSTANCE_NAME)
    connected = socket.waitForConnected(500)
    socket.abort()
    return connected


class TrainWorker(QThread):
    """Runs the trainer in the background so the window stays responsive."""
    line = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, args, threads, parent=None):
        super().__init__(parent)
        self.args = args
        self.threads = threads

    def run(self):
        command = build_command(self.args, self.threads)
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


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesseract Font Trainer")
        self.resize(1120, 860)
        self.worker = None
        self.run_log = None
        self.run_stamp = None
        self.model_dir = None
        self.accepting = False  # True only while a run is shown; late output is ignored
        self.setAcceptDrops(True)

        self._build_controls()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._resources_panel())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._page())
        root.addWidget(scroll, 1)

        self.corpus_mode_changed(self.auto_corpus.isChecked())
        self.add_common_fonts()

    # ----- controls -----

    def _build_controls(self):
        # Fonts
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Font", "Weight", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setMinimumHeight(220)
        # Let drops reach the window, so any font dropped anywhere is added.
        self.table.setAcceptDrops(False)
        self.table.viewport().setAcceptDrops(False)
        self.add_font_button = QPushButton("Add font...")
        self.add_font_button.clicked.connect(self.browse_fonts)

        # Corpus
        self.auto_corpus = QCheckBox("On")
        self.auto_corpus.setChecked(True)
        self.auto_corpus.toggled.connect(self.corpus_mode_changed)
        self.generated_count = QSpinBox()
        self.generated_count.setRange(1, 1000)
        self.generated_count.setValue(DEFAULT_GENERATED_LINES)
        self.generated_count.setFixedWidth(110)
        self.corpus_edit = QPlainTextEdit()
        self.corpus_edit.setPlaceholderText(
            "One text line per row, at least 10. English or Latin script only.")
        self.corpus_edit.setAcceptDrops(False)
        self.corpus_edit.viewport().setAcceptDrops(False)
        self.corpus_edit.setMinimumHeight(200)
        self.load_corpus_button = QPushButton("Load text file...")
        self.load_corpus_button.clicked.connect(self.load_corpus)
        self.manual_box = QWidget()
        manual = QVBoxLayout(self.manual_box)
        manual.setContentsMargins(0, 0, 0, 0)
        manual.addWidget(self.corpus_edit)
        manual_buttons = QHBoxLayout()
        manual_buttons.addWidget(self.load_corpus_button)
        manual_buttons.addStretch()
        manual.addLayout(manual_buttons)

        # Training
        self.name = QLineEdit("minecraft")
        self.name.setFixedWidth(160)
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 5000)
        self.iterations.setValue(DEFAULT_ITERATIONS)
        self.iterations.setFixedWidth(110)
        self.font_size = QSpinBox()
        self.font_size.setRange(16, 96)
        self.font_size.setValue(DEFAULT_FONT_SIZE)
        self.font_size.setFixedWidth(110)

        # Resources
        self.memory_gb = QSpinBox()
        self.memory_gb.setRange(4, 128)
        self.memory_gb.setValue(DEFAULT_MEMORY_GB)
        self.memory_gb.setFixedWidth(110)
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, os.cpu_count() or 1)
        self.cpu_threads.setValue(min(DEFAULT_THREADS, os.cpu_count() or 1))
        self.cpu_threads.setFixedWidth(110)

        # Run, results and log
        self.primary = QPushButton("adjust the font")
        self.primary.setObjectName("primary")
        self.primary.setMinimumWidth(220)
        self.primary.clicked.connect(self.on_primary)
        self.status = QLabel("Ready. Check the fonts and settings, then press adjust the font.")
        self.status.setWordWrap(True)
        self.current_label = QLabel("Current font: not started")
        self.current_bar = QProgressBar()
        self.overall_label = QLabel("All fonts: not started")
        self.overall_bar = QProgressBar()
        for bar in (self.current_bar, self.overall_bar):
            bar.setRange(0, 100)
            bar.setValue(0)

        self.results = QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(
            ["Font", "Before", "After", "CER before", "CER after"])
        self.results.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.results.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeToContents)
        self.results.verticalHeader().setVisible(False)
        self.results.setSelectionMode(QAbstractItemView.NoSelection)
        self.results.setMinimumHeight(120)
        self.open_model_button = QPushButton("Open model folder")
        self.open_model_button.clicked.connect(self.open_model_folder)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress will appear here.")
        self.log.setMinimumHeight(260)
        self.open_logs_button = QPushButton("Open logs folder")
        self.open_logs_button.clicked.connect(self.open_logs)

    # ----- layout -----

    def _resources_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(18)
        title = QLabel("Resources")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        layout.addLayout(self._setting(
            "Memory limit", "Most memory WSL may use.", self.memory_gb))
        layout.addLayout(self._setting(
            "CPU threads", "Most processor threads training may use.", self.cpu_threads))
        gpu = QLabel("Not used")
        gpu.setObjectName("cardDesc")
        layout.addLayout(self._setting(
            "GPU", "Tesseract training runs on the CPU only.", gpu))
        note = QLabel("Memory and CPU limits apply the next time you press adjust the font. "
                      "If they change, WSL restarts first.")
        note.setObjectName("cardDesc")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        return panel

    def _page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        head = QLabel("Tesseract Font Trainer")
        head.setObjectName("pageTitle")
        sub = QLabel("Teach Tesseract the fonts you use, then check how much it improved.")
        sub.setObjectName("pageDesc")
        titles.addWidget(head)
        titles.addWidget(sub)
        header.addLayout(titles, 1)
        header.addWidget(self.primary, 0, Qt.AlignVCenter)
        layout.addLayout(header)

        banner = QFrame()
        banner.setObjectName("banner")
        banner_layout = QVBoxLayout(banner)
        banner_layout.setContentsMargins(16, 12, 16, 12)
        banner_layout.addWidget(self.status)
        layout.addWidget(banner)

        progress_body = QVBoxLayout()
        progress_body.addWidget(self.current_label)
        progress_body.addWidget(self.current_bar)
        progress_body.addWidget(self.overall_label)
        progress_body.addWidget(self.overall_bar)
        layout.addWidget(self._card(
            "Progress", "Current font shows the step being done now. All fonts shows the whole run.",
            progress_body))

        fonts_body = QVBoxLayout()
        fonts_body.addWidget(self.table)
        font_buttons = QHBoxLayout()
        font_buttons.addWidget(self.add_font_button)
        font_buttons.addStretch()
        fonts_body.addLayout(font_buttons)
        layout.addWidget(self._card(
            "Fonts",
            "Drop font files anywhere in this window. Weight = how many times each training "
            "line is rendered in that font. Common Windows fonts are added at startup.",
            fonts_body))

        corpus_body = QVBoxLayout()
        corpus_body.addLayout(self._setting(
            "Generate lines automatically",
            "A new set of random Minecraft and mod-loader lines is made for every run. "
            "Turn this off to use your own text.",
            self.auto_corpus))
        corpus_body.addLayout(self._setting(
            "Lines per run", "Only used when generation is on.", self.generated_count))
        corpus_body.addWidget(self.manual_box)
        layout.addWidget(self._card(
            "Corpus",
            "The text lines the font is trained on. At least 10 lines are needed.",
            corpus_body))

        training_body = QVBoxLayout()
        training_body.addLayout(self._setting(
            "Model name", "Letters, digits, - or _. One name for the whole run.", self.name))
        training_body.addLayout(self._setting(
            "Iterations", "How long training runs. More takes longer.", self.iterations))
        training_body.addLayout(self._setting(
            "Font size", "Pixels for each rendered line.", self.font_size))
        layout.addWidget(self._card(
            "Training", None, training_body))

        results_body = QVBoxLayout()
        results_body.addWidget(self.results)
        results_buttons = QHBoxLayout()
        results_buttons.addWidget(self.open_model_button)
        results_buttons.addStretch()
        results_body.addLayout(results_buttons)
        layout.addWidget(self._card(
            "Results",
            "Exact matches on held-out lines, before and after training. Filled in when a "
            "run finishes.",
            results_body))

        log_body = QVBoxLayout()
        log_body.addWidget(self.log)
        log_buttons = QHBoxLayout()
        log_buttons.addWidget(self.open_logs_button)
        log_buttons.addStretch()
        log_body.addLayout(log_buttons)
        layout.addWidget(self._card(
            "Log", "Each run is also saved to data/logs.", log_body))

        layout.addStretch()
        return page

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
        """A settings row: text on the left, the control on the right."""
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

    def corpus_mode_changed(self, on):
        self.auto_corpus.setText("On" if on else "Off")
        self.generated_count.setEnabled(on)
        self.manual_box.setVisible(not on)

    def load_corpus(self):
        path, _ = QFileDialog.getOpenFileName(self, "Corpus", "", "Text (*.txt)")
        if path:
            self.corpus_edit.setPlainText(Path(path).read_text(encoding="utf-8"))
            self.auto_corpus.setChecked(False)

    def corpus_lines_for_run(self):
        """Generated lines when generation is on, otherwise the lines typed or loaded."""
        if self.auto_corpus.isChecked():
            return generate_lines(self.generated_count.value())
        return [l for l in self.corpus_edit.toPlainText().splitlines() if l.strip()]

    # ----- training -----

    def apply_wsl_limits(self):
        """Write the memory and CPU limits to .wslconfig. WSL restarts when they change.
        Returns False if the user cancels."""
        config = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".wslconfig"
        current = config.read_text(encoding="utf-8") if config.exists() else ""
        new = wslconfig_text(current, self.memory_gb.value(), self.cpu_threads.value())
        if new == current:
            return True
        answer = QMessageBox.question(
            self, "Restart WSL",
            "The memory or CPU limit changed. WSL must restart to apply it, which stops "
            "any WSL program that is running. Continue?")
        if answer != QMessageBox.Yes:
            return False
        config.write_text(new, encoding="utf-8")
        subprocess.run(["wsl", "--shutdown"], creationflags=CREATE_NO_WINDOW)
        return True

    def start_training(self):
        problem = self.validate()
        if problem:
            QMessageBox.warning(self, "Cannot start", problem)
            return
        if not self.apply_wsl_limits():
            return

        name = self.name.text().strip()
        self.accepting = True
        self.log.clear()
        self.results.setRowCount(0)
        self.current_bar.setValue(0)
        self.overall_bar.setValue(0)
        self.current_label.setText("Current font: starting")
        self.overall_label.setText("All fonts: starting")
        self.model_dir = None
        self.open_run_log(name)
        self.set_running(True)

        lines = self.corpus_lines_for_run()
        corpus_text = "\n".join(lines) + "\n"
        corpus_file = LOG_DIR / f"{self.run_stamp}_corpus.txt"
        corpus_file.write_text(corpus_text, encoding="utf-8")

        source = "generated automatically" if self.auto_corpus.isChecked() else "your own text"
        fonts = [{"path": to_wsl(path), "weight": weight} for path, weight in self.dataset()]
        args = [
            "--fonts-json", json.dumps(fonts),
            "--name", name,
            "--corpus-file", to_wsl(corpus_file),
            "--data-root", to_wsl(DATA / "runs" / self.run_stamp),
            "--iterations", str(self.iterations.value()),
            "--font-size", str(self.font_size.value()),
        ]

        self.record(f"Model: {name}")
        self.record(f"Iterations: {self.iterations.value()}, font size: {self.font_size.value()}px")
        self.record(f"Corpus: {len(lines)} lines, {source} "
                    f"(copy in logs: {self.run_stamp}_corpus.txt)")
        for path, weight in self.dataset():
            self.record(f"Font: {path} (weight {weight})")
        self.status.setText("Training... this can take a long time. Keep this window open.")
        self.worker = TrainWorker(args, self.cpu_threads.value())
        self.worker.line.connect(self.record)
        self.worker.progress.connect(self.show_progress)
        self.worker.done.connect(self.show_results)
        self.worker.failed.connect(self.show_failure)
        self.worker.start()

    def on_primary(self):
        if self.worker is not None and self.worker.isRunning():
            if not self.worker.stopped:
                self.stop_training()
            return
        self.start_training()

    def stop_training(self):
        if self.worker is None or not self.worker.isRunning():
            return
        self.status.setText("Stopping...")
        self.worker.stop()
        self.finish_stopped()

    def finish_stopped(self):
        """Close the run now. The trainer's remaining output is ignored from here on."""
        self.record("interrupted by a user")
        if self.run_log:
            self.run_log.close()
            self.run_log = None
        self.accepting = False
        self.set_running(False)
        self.status.setText("Training stopped. Nothing was saved.")

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        event.accept()

    def set_running(self, on):
        self.primary.setText("Stop training" if on else "adjust the font")

    def show_progress(self, info):
        """Update both progress bars from a PROGRESS line of the trainer."""
        if not self.accepting:
            return
        total = max(info["total"], 1)
        overall = int(100 * info["done"] / total)
        count = max(info["count"], 1)
        current = int(100 * info["current"] / count)
        if info["stage"] == "prepare":
            step = f"Preparing lines {info['current']}/{info['count']}"
        elif info["stage"] == "train":
            step = f"Training, iteration {info['current']}/{info['count']}"
        else:
            step = f"Scoring on held-out lines ({info['current']}/{info['count']})"
        self.current_label.setText(f"Current font {info['label']}: {step}")
        self.current_bar.setValue(current)
        self.overall_label.setText(f"All fonts: {info['done']}/{info['total']} steps")
        self.overall_bar.setValue(overall)

    def validate(self):
        if self.table.rowCount() == 0:
            return "Add at least one font to the dataset."
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", self.name.text().strip()):
            return "Model name: use letters, digits, - or _ (max 40 characters)."
        if not self.auto_corpus.isChecked():
            lines = [l for l in self.corpus_edit.toPlainText().splitlines() if l.strip()]
            if len(lines) < 10:
                return f"Your own corpus needs at least 10 lines. It has {len(lines)}."
        elif self.generated_count.value() < 10:
            return "Lines per run must be at least 10. Training needs 10 lines to split."
        return None

    # ----- run log -----

    def open_run_log(self, name):
        """Every run gets a text log and a JSON result in data/logs, so you can read them later."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.run_stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        self.run_log = open(LOG_DIR / f"{self.run_stamp}.log", "w", encoding="utf-8")
        self.record(f"Run started: {self.run_stamp}")

    def record(self, text):
        if not self.accepting:
            return
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
        self._open_folder(LOG_DIR)

    def open_model_folder(self):
        if self.model_dir and self.model_dir.exists():
            self._open_folder(self.model_dir)
        else:
            self.status.setText("No trained model yet. Finish a run first.")

    def _open_folder(self, folder):
        if hasattr(os, "startfile"):
            os.startfile(folder)
        else:
            self.status.setText(f"Folder: {folder}")

    # ----- results -----

    def show_results(self, result):
        if not self.accepting:
            return
        self.set_running(False)
        (LOG_DIR / f"{self.run_stamp}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")

        self.results.setRowCount(0)
        lines = ["", "Results (exact matches on held-out lines):"]
        for label, scores in result["scores"].items():
            b, t = scores["baseline"], scores["trained"]
            font = label.split("_", 1)[-1]
            self._add_result_row(font, b, t)
            lines.append(
                f"  {font}: before {b['exact']}/{b['total']} (CER {b['cer']}), "
                f"after {t['exact']}/{t['total']} (CER {t['cer']})")

        self.model_dir = DATA / "runs" / self.run_stamp / result["name"] / "model"
        model = self.model_dir / f"{result['name']}.traineddata"
        lines.append(f"Model saved to: {model}")
        self.record("\n".join(lines))
        self.close_run_log()
        self.accepting = False
        self.status.setText("Done. The model is saved. Results are in the table below.")

    def _add_result_row(self, font, before, after):
        row = self.results.rowCount()
        self.results.insertRow(row)
        values = [font, f"{before['exact']}/{before['total']}",
                  f"{after['exact']}/{after['total']}",
                  str(before["cer"]), str(after["cer"])]
        for column, value in enumerate(values):
            self.results.setItem(row, column, QTableWidgetItem(value))

    def show_failure(self, detail):
        if not self.accepting:
            return
        self.set_running(False)
        if detail == "Training stopped.":
            self.finish_stopped()
            return
        self.status.setText("Training failed. See the log below.")
        self.record(f"\nFAILED: {detail}")
        self.close_run_log()
        self.accepting = False


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(13)
    app.setFont(font)
    app.setStyleSheet(stylesheet(DARK if system_is_dark() else LIGHT))

    if another_copy_is_open():
        QMessageBox.information(None, "Tesseract Font Trainer",
                                "The trainer is already open. Use that window.")
        return
    QLocalServer.removeServer(SINGLE_INSTANCE_NAME)
    instance_server = QLocalServer()
    instance_server.listen(SINGLE_INSTANCE_NAME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
