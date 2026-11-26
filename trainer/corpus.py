"""Split a text corpus into training and control lines.

Each line appears in exactly one set, so the control score measures
reading ability on unseen text rather than memorized text.

Only lines the English model can use are kept. Cyrillic lines are dropped,
because the English model has no Cyrillic characters to learn from.
Russian is handled by Tesseract's separate 'rus' model.
"""

import random
import unicodedata


def has_cyrillic(text: str) -> bool:
    return any("CYRILLIC" in unicodedata.name(ch, "") for ch in text)


def split_corpus(text: str, control_fraction: float = 0.2, seed: int = 1234):
    """Return (train_lines, control_lines, dropped_count) from a multi-line corpus."""
    seen = set()
    lines = []
    dropped = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        if has_cyrillic(line):
            dropped += 1
            continue
        lines.append(line)

    if len(lines) < 10:
        raise ValueError(
            f"Need at least 10 unique English or Latin-script lines, got {len(lines)}. "
            f"Dropped {dropped} Cyrillic line(s)."
        )

    rng = random.Random(seed)
    rng.shuffle(lines)
    control_count = max(1, round(len(lines) * control_fraction))
    control = lines[:control_count]
    train = lines[control_count:]
    return train, control, dropped
