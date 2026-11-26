"""Local web interface for training Tesseract on a custom font.

Run it, then open http://localhost:8000 in a browser.
Upload a font, paste a corpus of text lines, and start training. The page
shows a log while training runs, then before-and-after accuracy on held-out lines.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from trainer.pipeline import train_font

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FONTS = DATA / "fonts"
RUNS = DATA / "runs"
PORT = 8000

state = {"running": False, "log": [], "result": None, "error": None}
state_lock = threading.Lock()


def log(message):
    with state_lock:
        state["log"].append(str(message))


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value or ""):
        raise ValueError("Use letters, digits, - or _ (max 40 characters) for names.")
    return value


def run_training(job):
    try:
        fonts = [(str(FONTS / job["font"]), 1)] + [
            (str(FONTS / f), 1) for f in job.get("control_fonts", [])
        ]
        result = train_font(
            name=job["name"],
            fonts=fonts,
            corpus=job["corpus"],
            data_root=RUNS,
            log=log,
            font_size=int(job.get("font_size", 32)),
            iterations=int(job.get("iterations", 400)),
        )
        with state_lock:
            state["result"] = result
    except Exception as error:  # surface any failure in the page
        with state_lock:
            state["error"] = str(error)
        log(f"ERROR: {error}")
    finally:
        with state_lock:
            state["running"] = False


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Tesseract Font Trainer</title>
<style>
body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }
textarea { width: 100%; height: 220px; font-family: monospace; }
input[type=text], input[type=number] { width: 220px; }
pre { background: #f4f4f4; padding: 1em; white-space: pre-wrap; max-height: 320px; overflow: auto; }
table { border-collapse: collapse; margin-top: 1em; }
td, th { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
.card { border: 1px solid #ddd; border-radius: 8px; padding: 1em; margin-bottom: 1em; }
</style></head><body>
<h1>Tesseract Font Trainer</h1>

<div class="card">
<h3>1. Upload a font</h3>
<input type="file" id="fontfile" accept=".ttf,.otf">
<button onclick="uploadFont()">Upload</button>
<span id="uploadStatus"></span>
<p>Uploaded fonts: <span id="fontList">none</span></p>
</div>

<div class="card">
<h3>2. Corpus (one text line per row, at least 10)</h3>
<textarea id="corpus" placeholder="The Mod File has mods that were not found&#10;D:\\Games\\MultiMC\\instances\\1.20.1\\.minecraft\\mods\\glassseam-fix.jar&#10;Файл модов не найден"></textarea>
</div>

<div class="card">
<h3>3. Settings and train</h3>
<p>Font to train: <input type="text" id="font" placeholder="myfont.ttf"></p>
<p>Model name: <input type="text" id="name" placeholder="minecraft"></p>
<p>Font size (px): <input type="number" id="font_size" value="32"></p>
<p>Training iterations: <input type="number" id="iterations" value="400"></p>
<p>Extra control fonts (for forgetting check, uploaded names, comma-separated):
<input type="text" id="control_fonts" placeholder="arial.ttf, consola.ttf"></p>
<button onclick="startTraining()">Start training</button>
<span id="trainStatus"></span>
</div>

<div class="card">
<h3>Progress</h3>
<pre id="log">Idle.</pre>
<div id="results"></div>
</div>

<script>
async function refreshFonts() {
  const r = await fetch('/fonts');
  const names = await r.json();
  document.getElementById('fontList').textContent = names.length ? names.join(', ') : 'none';
}
async function uploadFont() {
  const file = document.getElementById('fontfile').files[0];
  if (!file) { alert('Choose a font file first.'); return; }
  const r = await fetch('/upload?name=' + encodeURIComponent(file.name), { method: 'POST', body: file });
  document.getElementById('uploadStatus').textContent = r.ok ? 'Uploaded.' : await r.text();
  refreshFonts();
}
async function startTraining() {
  const controlFonts = document.getElementById('control_fonts').value
    .split(',').map(s => s.trim()).filter(s => s);
  const body = {
    font: document.getElementById('font').value.trim(),
    name: document.getElementById('name').value.trim(),
    corpus: document.getElementById('corpus').value,
    font_size: document.getElementById('font_size').value,
    iterations: document.getElementById('iterations').value,
    control_fonts: controlFonts,
  };
  const r = await fetch('/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  document.getElementById('trainStatus').textContent = r.ok ? 'Started.' : await r.text();
  pollStatus();
}
async function pollStatus() {
  const r = await fetch('/status');
  const s = await r.json();
  document.getElementById('log').textContent = s.log.join('\\n') || 'Idle.';
  let html = '';
  if (s.error) html += '<p style="color:#b00">Error: ' + s.error + '</p>';
  if (s.result) {
    html += '<h4>Results (exact matches on held-out lines)</h4><table><tr><th>Font</th><th>Baseline</th><th>Trained</th></tr>';
    for (const [label, v] of Object.entries(s.result.scores)) {
      html += '<tr><td>' + label + '</td><td>' + v.baseline.exact + '/' + v.baseline.total +
              ' (CER ' + v.baseline.cer + ')</td><td>' + v.trained.exact + '/' + v.trained.total +
              ' (CER ' + v.trained.cer + ')</td></tr>';
    }
    html += '</table><p>Model saved to: ' + s.result.model + '</p>';
  }
  document.getElementById('results').innerHTML = html;
  if (s.running) setTimeout(pollStatus, 2000);
}
refreshFonts();
pollStatus();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/fonts":
            FONTS.mkdir(parents=True, exist_ok=True)
            names = sorted(p.name for p in FONTS.iterdir() if p.suffix.lower() in (".ttf", ".otf"))
            self._send(200, names)
        elif path == "/status":
            with state_lock:
                self._send(200, dict(state, log=list(state["log"])))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if path == "/upload":
            try:
                query = parse_qs(urlparse(self.path).query)
                filename = Path(query.get("name", [""])[0]).name
                if Path(filename).suffix.lower() not in (".ttf", ".otf"):
                    raise ValueError("Only .ttf and .otf files are supported.")
                FONTS.mkdir(parents=True, exist_ok=True)
                (FONTS / filename).write_bytes(body)
                self._send(200, {"saved": filename})
            except ValueError as error:
                self._send(400, {"error": str(error)})
            return

        if path == "/train":
            try:
                job = json.loads(body or b"{}")
                job["name"] = safe_name(job.get("name", ""))
                if not (FONTS / Path(job.get("font", "")).name).exists():
                    raise ValueError("Choose an uploaded font to train.")
                job["font"] = Path(job["font"]).name
                if len(job.get("corpus", "").strip().splitlines()) < 10:
                    raise ValueError("Corpus needs at least 10 lines.")
            except (ValueError, json.JSONDecodeError) as error:
                self._send(400, {"error": str(error)})
                return
            with state_lock:
                if state["running"]:
                    self._send(409, {"error": "Training is already running."})
                    return
                state.update(running=True, log=[], result=None, error=None)
            threading.Thread(target=run_training, args=(job,), daemon=True).start()
            self._send(200, {"started": job["name"]})
            return

        self._send(404, {"error": "not found"})

    def log_message(self, format, *args):
        return  # keep the console quiet; progress is shown in the page


if __name__ == "__main__":
    FONTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Open http://localhost:{PORT} in your browser. Press Ctrl+C to stop.")
    server.serve_forever()
