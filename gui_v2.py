"""Tesseract Font Trainer, second window design.

Runs on Windows. Training runs inside WSL, because Tesseract's training tools are
Linux programs. This file is a new window written from scratch. It uses the same
trainer (trainer/cli.py) and the same data folders as the first window (gui.py).

Pages (left side): Training, Fonts, Corpus, Results, Logs, Settings.

Only one copy of the window runs at a time. Double-click gui_v2.pyw to open it
without a console window.

Requirements (Windows): pip install -r requirements-gui.txt
"""

# Software rendering avoids a crash with the graphics driver on this PC. It must be
# set before PySide6 loads.
import os

os.environ.setdefault("QT_OPENGL", "software")

import json
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    import winreg
except ImportError:  # not on Windows, so the theme cannot be read
    winreg = None

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QThread, QTimer, Qt, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"
LOG_DIR = DATA / "logs"
RUNS_DIR = DATA / "runs"
sys.path.insert(0, str(REPO / "tools"))
from make_corpus import generate  # noqa: E402  (the Minecraft line generator)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WINDOWS_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
COMMON_FONTS = [
    "arial.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "georgia.ttf",
    "tahoma.ttf", "calibri.ttf", "consola.ttf", "segoeui.ttf",
]
PAGES = ["Training", "Fonts", "Corpus", "Results", "Logs", "Settings"]
INSTANCE_NAME = "TesseractFontTrainer"  # same name as the first window: one copy at a time
DEFAULT_ITERATIONS = 400
DEFAULT_FONT_SIZE = 32
DEFAULT_GENERATED_LINES = 200
DEFAULT_MEMORY_GB = 16
DEFAULT_THREADS = 2
MIN_LINES = 10
BAR_STEPS = 1000  # progress bars move in tenths of a percent

LIGHT = dict(bg="#F3F3F3", nav="#EBEBEB", nav_sel="#FFFFFF", card="#FFFFFF",
             border="#E2E2E2", hover="#F2F2F2", text="#1B1B1B", sub="#5F5F5F",
             accent="#0067C0", on_accent="#FFFFFF", danger="#C42B1C",
             banner="#E8F1FB", field="#FFFFFF")
DARK = dict(bg="#1F1F1F", nav="#191919", nav_sel="#2C2C2C", card="#2B2B2B",
            border="#3A3A3A", hover="#353535", text="#FFFFFF", sub="#B8B8B8",
            accent="#4CC2FF", on_accent="#000000", danger="#FF6B6B",
            banner="#253A4E", field="#222222")


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
QWidget {{ background: {t['bg']}; color: {t['text']}; font-family: "Segoe UI"; font-size: 10pt; }}
QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 22pt; font-weight: 600; }}
QLabel#subtitle {{ color: {t['sub']}; font-size: 11pt; }}
QLabel#cardTitle {{ font-size: 12pt; font-weight: 600; }}
QLabel#hint {{ color: {t['sub']}; }}
QLabel#banner {{ background: {t['banner']}; border-radius: 8px; padding: 12px 16px; }}
QFrame#card {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 10px; }}
QListWidget#nav {{ background: {t['nav']}; border: none; padding: 12px 8px; outline: none; }}
QListWidget#nav::item {{ padding: 10px 14px; border-radius: 6px; margin: 2px 0; }}
QListWidget#nav::item:selected {{ background: {t['nav_sel']}; }}
QListWidget#nav::item:hover {{ background: {t['hover']}; }}
QListWidget {{ background: {t['field']}; border: 1px solid {t['border']}; border-radius: 6px; }}
QPushButton {{ background: {t['card']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 7px 16px; }}
QPushButton:hover {{ background: {t['hover']}; }}
QPushButton:disabled {{ color: {t['sub']}; }}
QPushButton#primary {{ background: {t['accent']}; color: {t['on_accent']}; border: none; font-weight: 600; padding: 12px 28px; }}
QPushButton#primary[stop="true"] {{ background: {t['danger']}; color: #FFFFFF; }}
QPushButton#primary:disabled {{ background: {t['border']}; color: {t['sub']}; }}
QLineEdit, QPlainTextEdit, QSpinBox {{ background: {t['field']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 5px 8px; }}
QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus {{ border: 1px solid {t['accent']}; }}
QTableWidget {{ background: {t['field']}; border: 1px solid {t['border']}; border-radius: 6px; gridline-color: {t['border']}; }}
QHeaderView::section {{ background: {t['card']}; color: {t['sub']}; border: none; border-bottom: 1px solid {t['border']}; padding: 6px; font-weight: 600; }}
QProgressBar {{ background: {t['field']}; border: 1px solid {t['border']}; border-radius: 6px; text-align: center; color: {t['text']}; min-height: 20px; }}
QProgressBar::chunk {{ background: {t['accent']}; border-radius: 5px; }}
QCheckBox {{ spacing: 8px; }}
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
    """The WSL command that runs the trainer inside the repo folder. The CPU thread limit
    is passed as OMP_THREAD_LIMIT, which Tesseract's OpenMP code reads."""
    return ["wsl", "--cd", to_wsl(REPO), "--", "env", f"OMP_THREAD_LIMIT={threads}",
            "python3", "-m", "trainer.cli", *args]


def clean_line(raw):
    """Some WSL output arrives as UTF-16, which leaves a NUL after every character."""
    return raw.replace("\x00", "").rstrip()


def wslconfig_text(text, memory_gb, processors):
    """Set memory and processors under [wsl2] and keep every other setting."""
    sections = {"": []}
    order = [""]
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip().lower()
            if current not in sections:
                sections[current] = []
                order.append(current)
            continue
        sections[current].append(line)
    kept = [line for line in sections.get("wsl2", [])
            if line.split("=")[0].strip().lower() not in ("memory", "processors")]
    sections["wsl2"] = kept + [f"memory={memory_gb}GB", f"processors={processors}"]
    if "wsl2" not in order:
        order.append("wsl2")
    out = []
    for name in order:
        if name:
            out.append(f"[{name}]")
        out.extend(sections[name])
    return "\n".join(out).strip() + "\n"


def open_path(path):
    """Open a file or folder with the Windows default program."""
    if hasattr(os, "startfile"):
        os.startfile(str(path))
    else:
        subprocess.Popen(["xdg-open", str(path)])


class TrainWorker(QThread):
    """Runs the trainer in WSL and reads its output. Output lines go to the window;
    PROGRESS lines drive the bars, and the RESULT line carries the final scores."""

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
        """Stop the trainer and the programs it started. They run inside WSL, so they are
        killed there. Killing only the Windows side would leave them running."""
        self.stopped = True
        try:
            subprocess.Popen(
                ["wsl", "--", "pkill", "-f", "trainer.cli|lstmtraining|text2image|tesseract"],
                creationflags=CREATE_NO_WINDOW)
        except OSError:
            pass
        if self.process is not None:
            try:
                self.process.kill()
            except OSError:
                pass

    def run(self):
        # Any error must reach the window, so the window never waits for a thread that ended.
        try:
            self._run_trainer()
        except Exception:
            self.failed.emit("Window error:\n" + traceback.format_exc())

    def _run_trainer(self):
        command = build_command(self.args, self.threads)
        self.line.emit("Starting the trainer in WSL. The first output can take a little while.")
        try:
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=CREATE_NO_WINDOW,
            )
        except FileNotFoundError:
            self.failed.emit("WSL was not found. Install WSL and Ubuntu first (see README).")
            return
        self.line.emit(f"WSL started (process {self.process.pid}), waiting for the trainer.")

        result = None
        errors = []
        for raw in self.process.stdout:
            text = clean_line(raw)
            if text.startswith("PROGRESS:"):
                try:
                    self.progress.emit(json.loads(text[len("PROGRESS:"):]))
                except ValueError:
                    pass
            elif text.startswith("RESULT:"):
                try:
                    result = json.loads(text[len("RESULT:"):])
                except ValueError:
                    errors.append("Could not read the result line.")
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


def another_copy_is_open():
    socket = QLocalSocket()
    socket.connectToServer(INSTANCE_NAME)
    connected = socket.waitForConnected(300)
    socket.abort()
    return connected


def make_bar():
    bar = QProgressBar()
    bar.setRange(0, BAR_STEPS)
    bar.setValue(0)
    return bar


def card(title=None, hint=None):
    """A rounded panel. Returns the frame and its layout."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)
    if title:
        label = QLabel(title)
        label.setObjectName("cardTitle")
        layout.addWidget(label)
    if hint:
        label = QLabel(hint)
        label.setObjectName("hint")
        label.setWordWrap(True)
        layout.addWidget(label)
    return frame, layout


def page_header(title, subtitle):
    box = QVBoxLayout()
    box.setSpacing(2)
    heading = QLabel(title)
    heading.setObjectName("title")
    text = QLabel(subtitle)
    text.setObjectName("subtitle")
    text.setWordWrap(True)
    box.addWidget(heading)
    box.addWidget(text)
    return box


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tesseract Font Trainer")
        self.resize(1120, 760)
        self.setMinimumSize(920, 640)
        self.setAcceptDrops(True)

        self.fonts = []  # list of {"path": str, "weight": int}
        self.worker = None
        self.running = False
        self.run_stamp = None
        self.run_log = None
        self.model_dir = None
        self.last_output = 0.0
        self.run_started = 0.0
        self.animations = {}

        self.heartbeat = QTimer(self)
        self.heartbeat.timeout.connect(self.heartbeat_tick)

        self.build_ui()
        self.add_common_fonts()
        self.refresh_logs()
        self.update_corpus_state()

    # ----- layout -----

    def build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(root)

        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(200)
        for name in PAGES:
            self.nav.addItem(QListWidgetItem(name))

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_training_page())
        self.stack.addWidget(self.build_fonts_page())
        self.stack.addWidget(self.build_corpus_page())
        self.stack.addWidget(self.build_results_page())
        self.stack.addWidget(self.build_logs_page())
        self.stack.addWidget(self.build_settings_page())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(28, 24, 28, 24)
        content_layout.addWidget(self.stack)

        layout.addWidget(self.nav)
        layout.addWidget(content, 1)

    def build_training_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.addLayout(page_header(
            "Training", "Teach Tesseract the fonts you use, then check how much it improved."))
        header.addStretch(1)
        self.primary = QPushButton("Start training")
        self.primary.setObjectName("primary")
        self.primary.clicked.connect(self.on_primary)
        header.addWidget(self.primary, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(header)

        self.status = QLabel("Ready. Add fonts on the Fonts page, then press Start training.")
        self.status.setObjectName("banner")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        settings, form_layout = card("Run")
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        self.name = QLineEdit("minecraft")
        self.name.setPlaceholderText("letters, digits, - or _")
        self.iterations = QSpinBox()
        self.iterations.setRange(1, 5000)
        self.iterations.setValue(DEFAULT_ITERATIONS)
        self.font_size = QSpinBox()
        self.font_size.setRange(16, 96)
        self.font_size.setValue(DEFAULT_FONT_SIZE)
        form.addRow("Model name", self.name)
        form.addRow("Training iterations", self.iterations)
        form.addRow("Font size (px)", self.font_size)
        form_layout.addLayout(form)
        layout.addWidget(settings)

        progress, progress_layout = card("Progress")
        self.current_label = QLabel("Current font: not started")
        self.current_bar = make_bar()
        self.overall_label = QLabel("All fonts: not started")
        self.overall_bar = make_bar()
        self.render_label = QLabel()
        self.render_bar = make_bar()
        self.render_label.hide()
        self.render_bar.hide()
        for widget in (self.current_label, self.current_bar, self.overall_label,
                       self.overall_bar, self.render_label, self.render_bar):
            progress_layout.addWidget(widget)
        layout.addWidget(progress)

        live, live_layout = card("Live log", "Every step is written to data/logs as it happens.")
        self.live_log = QPlainTextEdit()
        self.live_log.setReadOnly(True)
        self.live_log.setMinimumHeight(160)
        live_layout.addWidget(self.live_log)
        layout.addWidget(live, 1)
        return page

    def build_fonts_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addLayout(page_header(
            "Fonts",
            "The fonts Tesseract trains on. Drop font files anywhere in this window, "
            "or add them with the buttons below."))

        frame, frame_layout = card(
            None, "Weight is how many times each training line is rendered in that font. "
                  "A higher weight means the model sees that font more often.")
        self.font_table = QTableWidget(0, 3)
        self.font_table.setHorizontalHeaderLabels(["Font", "Weight", ""])
        self.font_table.verticalHeader().setVisible(False)
        self.font_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.font_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.font_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.font_table.setColumnWidth(1, 110)
        self.font_table.setColumnWidth(2, 110)
        frame_layout.addWidget(self.font_table)

        buttons = QHBoxLayout()
        add = QPushButton("Add font...")
        add.clicked.connect(self.choose_fonts)
        common = QPushButton("Add common Windows fonts")
        common.clicked.connect(self.add_common_fonts)
        buttons.addWidget(add)
        buttons.addWidget(common)
        buttons.addStretch(1)
        frame_layout.addLayout(buttons)
        layout.addWidget(frame, 1)
        return page

    def build_corpus_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addLayout(page_header(
            "Corpus",
            "The text lines the fonts are trained on. About 20% of the lines are kept "
            "aside as a check and never trained on."))

        frame, frame_layout = card()
        self.auto_corpus = QCheckBox("Generate lines automatically for each run")
        self.auto_corpus.setChecked(True)
        self.auto_corpus.toggled.connect(self.update_corpus_state)
        frame_layout.addWidget(self.auto_corpus)

        form = QFormLayout()
        self.generated_count = QSpinBox()
        self.generated_count.setRange(MIN_LINES, 5000)
        self.generated_count.setValue(DEFAULT_GENERATED_LINES)
        self.generated_count.valueChanged.connect(self.update_corpus_state)
        form.addRow("Lines per run", self.generated_count)
        frame_layout.addLayout(form)
        frame_layout.addWidget(QLabel(
            "Each run gets a new random set of Minecraft and mod-loader lines. "
            "A copy is saved next to the run's log."))
        layout.addWidget(frame)

        custom, custom_layout = card(
            "Your own text", "Turn off automatic lines to type or load your own. "
                             "One line per row, at least 10 lines.")
        self.corpus_edit = QPlainTextEdit()
        self.corpus_edit.setPlaceholderText("One line per row.")
        self.corpus_edit.textChanged.connect(self.update_corpus_state)
        custom_layout.addWidget(self.corpus_edit)
        row = QHBoxLayout()
        self.load_corpus = QPushButton("Load text file...")
        self.load_corpus.clicked.connect(self.load_corpus_file)
        row.addWidget(self.load_corpus)
        row.addStretch(1)
        self.corpus_count = QLabel()
        row.addWidget(self.corpus_count)
        custom_layout.addLayout(row)
        layout.addWidget(custom, 1)
        return page

    def build_results_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addLayout(page_header(
            "Results", "Exact matches on the held-out lines, before and after training."))

        frame, frame_layout = card()
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(
            ["Font", "Before", "After", "Error rate before", "Error rate after"])
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        frame_layout.addWidget(self.results_table)

        row = QHBoxLayout()
        self.model_label = QLabel("No model yet.")
        self.model_label.setObjectName("hint")
        row.addWidget(self.model_label, 1)
        self.open_model = QPushButton("Open model folder")
        self.open_model.setEnabled(False)
        self.open_model.clicked.connect(self.open_model_folder)
        row.addWidget(self.open_model)
        frame_layout.addLayout(row)
        layout.addWidget(frame, 1)
        return page

    def build_logs_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addLayout(page_header(
            "Logs", "Every run is saved in data/logs: the log, the corpus used, and the result."))

        frame, frame_layout = card()
        self.log_list = QListWidget()
        self.log_list.itemDoubleClicked.connect(
            lambda item: open_path(item.data(Qt.ItemDataRole.UserRole)))
        frame_layout.addWidget(self.log_list)
        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_logs)
        open_log = QPushButton("Open log")
        open_log.clicked.connect(self.open_selected_log)
        open_folder = QPushButton("Open logs folder")
        open_folder.clicked.connect(lambda: (LOG_DIR.mkdir(parents=True, exist_ok=True),
                                             open_path(LOG_DIR)))
        row.addWidget(refresh)
        row.addWidget(open_log)
        row.addStretch(1)
        row.addWidget(open_folder)
        frame_layout.addLayout(row)
        layout.addWidget(frame, 1)
        return page

    def build_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(14)
        layout.addLayout(page_header(
            "Settings", "Limits for the training run. They apply the next time you start training."))

        frame, frame_layout = card("Resource limits")
        form = QFormLayout()
        self.memory_gb = QSpinBox()
        self.memory_gb.setRange(4, 128)
        self.memory_gb.setSuffix(" GB")
        self.memory_gb.setValue(DEFAULT_MEMORY_GB)
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(1, os.cpu_count() or 1)
        self.cpu_threads.setValue(min(DEFAULT_THREADS, os.cpu_count() or 1))
        gpu = QLabel("Not used. Tesseract training runs on the CPU only.")
        gpu.setObjectName("hint")
        form.addRow("Memory limit for WSL", self.memory_gb)
        form.addRow("CPU threads", self.cpu_threads)
        form.addRow("GPU", gpu)
        frame_layout.addLayout(form)
        frame_layout.addWidget(QLabel(
            "If the memory or CPU limit changes, WSL restarts before the run, which stops any "
            "WSL program that is running. The window asks first."))
        layout.addWidget(frame)
        layout.addStretch(1)
        return page

    # ----- fonts -----

    def add_font_path(self, path):
        path = str(path)
        if any(item["path"] == path for item in self.fonts):
            return
        self.fonts.append({"path": path, "weight": 1})
        self.refresh_font_table()

    def add_common_fonts(self):
        for name in COMMON_FONTS:
            path = WINDOWS_FONTS / name
            if path.exists():
                self.add_font_path(path)

    def choose_fonts(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add fonts", "", "Font files (*.ttf *.otf)")
        for path in paths:
            self.add_font_path(path)

    def remove_font(self, index):
        del self.fonts[index]
        self.refresh_font_table()

    def set_weight(self, index, value):
        self.fonts[index]["weight"] = value

    def refresh_font_table(self):
        self.font_table.setRowCount(len(self.fonts))
        for row, item in enumerate(self.fonts):
            self.font_table.setItem(row, 0, QTableWidgetItem(Path(item["path"]).name))
            weight = QSpinBox()
            weight.setRange(1, 10)
            weight.setValue(item["weight"])
            weight.valueChanged.connect(lambda value, index=row: self.set_weight(index, value))
            self.font_table.setCellWidget(row, 1, weight)
            remove = QPushButton("Remove")
            remove.clicked.connect(lambda checked=False, index=row: self.remove_font(index))
            self.font_table.setCellWidget(row, 2, remove)

    def dragEnterEvent(self, event):
        if any(url.toLocalFile().lower().endswith((".ttf", ".otf"))
               for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".ttf", ".otf")):
                self.add_font_path(path)

    # ----- corpus -----

    def update_corpus_state(self):
        auto = self.auto_corpus.isChecked()
        self.generated_count.setEnabled(auto)
        self.corpus_edit.setEnabled(not auto)
        self.load_corpus.setEnabled(not auto)
        if auto:
            self.corpus_count.setText(f"{self.generated_count.value()} lines will be generated")
        else:
            self.corpus_count.setText(f"{len(self.custom_lines())} lines")

    def custom_lines(self):
        return [line for line in self.corpus_edit.toPlainText().splitlines() if line.strip()]

    def load_corpus_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load text file", "", "Text files (*.txt)")
        if path:
            self.corpus_edit.setPlainText(Path(path).read_text(encoding="utf-8"))

    def corpus_lines_for_run(self):
        if self.auto_corpus.isChecked():
            return generate(self.generated_count.value(), random.randrange(1, 10**9))
        return self.custom_lines()

    # ----- resource limits -----

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
        if answer != QMessageBox.StandardButton.Yes:
            return False
        config.write_text(new, encoding="utf-8")
        subprocess.run(["wsl", "--shutdown"], creationflags=CREATE_NO_WINDOW, check=False)
        return True

    # ----- run -----

    def validate(self):
        if not self.fonts:
            return "Add at least one font on the Fonts page."
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", self.name.text().strip()):
            return "Model name: use letters, digits, - or _ (up to 40 characters)."
        if not self.auto_corpus.isChecked() and len(self.custom_lines()) < MIN_LINES:
            return f"Your own text needs at least {MIN_LINES} lines. It has {len(self.custom_lines())}."
        return None

    def on_primary(self):
        if self.running:
            self.stop_training()
        else:
            self.start_training()

    def start_training(self):
        if self.running:
            return
        problem = self.validate()
        if problem:
            QMessageBox.warning(self, "Cannot start", problem)
            return
        if not self.apply_wsl_limits():
            return

        name = self.name.text().strip()
        self.run_stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.run_log = open(LOG_DIR / f"{self.run_stamp}.log", "w", encoding="utf-8")
        lines = self.corpus_lines_for_run()
        corpus_file = LOG_DIR / f"{self.run_stamp}_corpus.txt"
        corpus_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        fonts_json = json.dumps([{"path": to_wsl(item["path"]), "weight": item["weight"]}
                                 for item in self.fonts])
        args = [
            "--fonts-json", fonts_json,
            "--name", name,
            "--corpus-file", to_wsl(corpus_file),
            "--data-root", to_wsl(RUNS_DIR / self.run_stamp),
            "--iterations", str(self.iterations.value()),
            "--font-size", str(self.font_size.value()),
        ]

        self.live_log.clear()
        self.results_table.setRowCount(0)
        self.model_dir = None
        self.open_model.setEnabled(False)
        self.model_label.setText("No model yet.")
        for bar in (self.current_bar, self.overall_bar, self.render_bar):
            self.reset_bar(bar)
        self.render_label.hide()
        self.render_bar.hide()
        self.set_label_text(self.current_label, "Current font: starting")
        self.set_label_text(self.overall_label, "All fonts: starting")

        self.running = True
        self.run_started = self.last_output = time.monotonic()
        self.heartbeat.start(15000)
        self.set_primary_mode(stop=True)
        self.status.setText("Training... this can take a long time. Keep this window open.")
        source = "generated automatically" if self.auto_corpus.isChecked() else "your own text"
        self.record(f"Run started: {self.run_stamp}")
        self.record(f"Model: {name}")
        self.record(f"Iterations: {self.iterations.value()}, font size: {self.font_size.value()}px")
        self.record(f"Corpus: {len(lines)} lines, {source} "
                    f"(copy in logs: {self.run_stamp}_corpus.txt)")
        for item in self.fonts:
            self.record(f"Font: {item['path']} (weight {item['weight']})")

        self.worker = TrainWorker(args, self.cpu_threads.value())
        self.worker.line.connect(self.on_line)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def stop_training(self):
        """Stop now. The window is updated at once; the WSL programs are stopped in the
        background, so the window never waits for WSL."""
        if not self.running:
            return
        self.record("interrupted by a user")
        self.finish_run()
        self.status.setText("Training stopped. Nothing was saved.")
        worker = self.worker
        if worker is not None:
            threading.Thread(target=worker.stop, daemon=True).start()

    def finish_run(self):
        """Close the run: the log is saved and the window goes back to idle."""
        self.running = False
        self.heartbeat.stop()
        self.set_primary_mode(stop=False)
        if self.run_log is not None:
            self.run_log.close()
            self.run_log = None
        self.refresh_logs()

    def set_primary_mode(self, stop):
        self.primary.setText("Stop training" if stop else "Start training")
        self.primary.setProperty("stop", "true" if stop else "false")
        self.primary.style().unpolish(self.primary)
        self.primary.style().polish(self.primary)

    def record(self, text):
        """Show a line in the live log and write it to the run's log with the time."""
        self.last_output = time.monotonic()
        self.live_log.appendPlainText(text)
        if self.run_log is not None:
            self.run_log.write(f"{datetime.now():%H:%M:%S} {text}\n")
            self.run_log.flush()

    def on_line(self, text):
        if self.running:
            self.record(text)

    def heartbeat_tick(self):
        """Log a line when the trainer has been quiet for 15 seconds, so the log never looks frozen."""
        if not self.running:
            return
        now = time.monotonic()
        if now - self.last_output >= 15:
            self.record(f"Still working: {int(now - self.run_started)} s since the start, "
                        f"{int(now - self.last_output)} s since the last line.")

    def on_progress(self, info):
        if not self.running:
            return
        if info["stage"] == "render":
            count = max(info["count"], 1)
            self.animate_bar(self.render_bar, info["current"] / count)
            self.set_label_text(self.render_label,
                                f"Rendering line {info['current']} of {info['count']} "
                                f"in {info['label']}")
            self.render_label.show()
            self.render_bar.show()
            return

        self.render_label.hide()
        self.render_bar.hide()
        total = max(info["total"], 1)
        count = max(info["count"], 1)
        if info["stage"] == "prepare":
            step = f"Preparing lines {info['current']}/{info['count']}"
        elif info["stage"] == "train":
            step = f"Training, iteration {info['current']}/{info['count']}"
        else:
            step = f"Checking held-out lines ({info['current']}/{info['count']})"
        self.set_label_text(self.current_label, f"Current font {info['label']}: {step}")
        self.set_label_text(self.overall_label,
                            f"All fonts: {info['done']}/{info['total']} steps")
        self.animate_bar(self.current_bar, info["current"] / count)
        self.animate_bar(self.overall_bar, info["done"] / total)

    def on_done(self, result):
        if not self.running:
            return
        lines = ["", "Results (exact matches on held-out lines):"]
        self.results_table.setRowCount(0)
        for label, scores in result["scores"].items():
            before, after = scores["baseline"], scores["trained"]
            font = label.split("_", 1)[-1]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            values = [font, f"{before['exact']}/{before['total']}",
                      f"{after['exact']}/{after['total']}",
                      str(before["cer"]), str(after["cer"])]
            for column, value in enumerate(values):
                self.results_table.setItem(row, column, QTableWidgetItem(value))
            lines.append(f"  {font}: before {before['exact']}/{before['total']} "
                         f"(CER {before['cer']}), after {after['exact']}/{after['total']} "
                         f"(CER {after['cer']})")

        name = result["name"]
        self.model_dir = RUNS_DIR / self.run_stamp / name / "model"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{self.run_stamp}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        lines.append(f"Model saved to: {self.model_dir / (name + '.traineddata')}")
        self.record("\n".join(lines))
        self.record(f"Log saved to: {LOG_DIR / (self.run_stamp + '.log')}")
        self.finish_run()
        self.model_label.setText(f"Model: {self.model_dir / (name + '.traineddata')}")
        self.open_model.setEnabled(True)
        self.status.setText("Done. The model is saved. See the Results page.")

    def on_failed(self, detail):
        if not self.running:
            return
        self.record(f"\nFAILED: {detail}")
        self.finish_run()
        self.status.setText("Training failed. See the live log.")

    def open_model_folder(self):
        if self.model_dir is not None:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            open_path(self.model_dir)

    def open_selected_log(self):
        item = self.log_list.currentItem()
        if item is not None:
            open_path(item.data(Qt.ItemDataRole.UserRole))

    def refresh_logs(self):
        self.log_list.clear()
        if not LOG_DIR.exists():
            return
        for path in sorted(LOG_DIR.glob("*.log"), reverse=True):
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.log_list.addItem(item)

    # ----- progress bars -----

    def set_label_text(self, label, text):
        label.setText(text)

    def animate_bar(self, bar, fraction):
        """Move a bar in small steps, so it never jumps. The time grows with the distance."""
        target = int(round(BAR_STEPS * min(max(fraction, 0.0), 1.0)))
        animation = self.animations.get(bar)
        if animation is None:
            animation = QPropertyAnimation(bar, b"value", self)
            animation.setEasingCurve(QEasingCurve.Type.Linear)
            self.animations[bar] = animation
        animation.stop()
        distance = abs(target - bar.value())
        animation.setDuration(max(80, min(1500, distance * 12)))
        animation.setStartValue(bar.value())
        animation.setEndValue(target)
        animation.start()

    def reset_bar(self, bar):
        animation = self.animations.get(bar)
        if animation is not None:
            animation.stop()
        bar.setValue(0)

    # ----- window events -----

    def closeEvent(self, event):
        if self.running:
            self.stop_training()
            if self.worker is not None:
                self.worker.wait(5000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(stylesheet(DARK if system_is_dark() else LIGHT))

    if another_copy_is_open():
        QMessageBox.information(None, "Already open",
                                "The Tesseract Font Trainer window is already open.")
        return
    QLocalServer.removeServer(INSTANCE_NAME)
    server = QLocalServer()
    server.listen(INSTANCE_NAME)

    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
