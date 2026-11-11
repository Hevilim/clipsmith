"""Parsing, slicing and writing of SRT / WebVTT subtitles."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})"
_TIMING_RE = re.compile(rf"{_TIME}\s*-->\s*{_TIME}")
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")

# Subtitles in the wild are often not UTF-8 (cp1251 is common for Russian releases).
_ENCODINGS = ("utf-8-sig", "cp1251", "latin-1")


@dataclass(frozen=True)
class Cue:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def format_timestamp(seconds: float, sep: str = ",") -> str:
    total_ms = max(0, round(seconds * 1000))
    h, rest = divmod(total_ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _clean(text: str) -> str:
    lines = (_TAG_RE.sub("", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def parse(text: str) -> list[Cue]:
    """Parse SRT or WebVTT content. Cues are re-numbered from 1 in time order."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw: list[tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            match = _TIMING_RE.search(line)
            if match:
                g = match.groups()
                start, end = _to_seconds(*g[:4]), _to_seconds(*g[4:])
                body = _clean("\n".join(lines[i + 1 :]))
                if body and end > start:
                    raw.append((start, end, body))
                break
    raw.sort(key=lambda c: c[0])
    return [Cue(i, s, e, t) for i, (s, e, t) in enumerate(raw, start=1)]


def load(path: str | Path) -> list[Cue]:
    data = Path(path).read_bytes()
    # UTF-16 decodes almost any byte string without error, so only trust it with a BOM.
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return parse(data.decode("utf-16"))
    for encoding in _ENCODINGS:
        try:
            return parse(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode subtitles: {path}")


def to_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{n}\n{format_timestamp(c.start)} --> {format_timestamp(c.end)}\n{c.text}\n"
        for n, c in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def slice_cues(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """Return cues overlapping [start, end], trimmed and shifted so the window starts at 0."""
    out = []
    for c in cues:
        if c.end <= start or c.start >= end:
            continue
        out.append(
            replace(c, start=max(c.start, start) - start, end=min(c.end, end) - start)
        )
    return out
