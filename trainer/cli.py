"""Command-line entry point for the trainer.

The Windows window calls this inside WSL. Progress lines go to stdout, and the
final result is printed as one line starting with 'RESULT:' followed by JSON.
"""

import argparse
import json
import sys
from pathlib import Path

from .pipeline import train_font


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune Tesseract on a set of fonts.")
    parser.add_argument("--fonts-json", required=True,
                        help='JSON list, e.g. [{"path": "/mnt/c/Fonts/a.ttf", "weight": 2}]')
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--corpus-file", required=True, help="Text file, one line per row")
    parser.add_argument("--data-root", required=True, help="Folder for run outputs")
    parser.add_argument("--iterations", type=int, default=400, help="Training iterations")
    parser.add_argument("--font-size", type=int, default=32, help="Rendered line size in px")
    args = parser.parse_args(argv)

    def log(message):
        print(message, flush=True)

    log(f"Trainer started: {args.name}, {args.iterations} iterations, "
        f"font size {args.font_size}px")

    try:
        fonts = [(item["path"], int(item["weight"])) for item in json.loads(args.fonts_json)]
        corpus = Path(args.corpus_file).read_text(encoding="utf-8")
        result = train_font(
            name=args.name,
            fonts=fonts,
            corpus=corpus,
            data_root=args.data_root,
            log=log,
            font_size=args.font_size,
            iterations=args.iterations,
        )
    except Exception as error:
        print(f"ERROR: {error}", flush=True)
        return 1

    print("RESULT:" + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
