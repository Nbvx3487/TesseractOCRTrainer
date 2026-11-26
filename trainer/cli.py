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
    parser = argparse.ArgumentParser(description="Fine-tune Tesseract on a font.")
    parser.add_argument("--font", required=True, help="Path to the .ttf or .otf file")
    parser.add_argument("--name", required=True, help="Model name")
    parser.add_argument("--corpus-file", required=True, help="Text file, one line per row")
    parser.add_argument("--data-root", required=True, help="Folder for run outputs")
    parser.add_argument("--font-size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--control-font", action="append", default=[],
                        help="Extra font for the forgetting check (repeatable)")
    args = parser.parse_args(argv)

    def log(message):
        print(message, flush=True)

    try:
        corpus = Path(args.corpus_file).read_text(encoding="utf-8")
        result = train_font(
            name=args.name,
            font_path=args.font,
            corpus=corpus,
            control_fonts=args.control_font,
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
