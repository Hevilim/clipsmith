"""Generate subtitles from speech with faster-whisper (optional dependency)."""

from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .subtitles import Cue
from .video import FFmpegError

# Close to large-v3 accuracy (good for Russian) at a fraction of the decoding cost.
DEFAULT_MODEL = "large-v3-turbo"

_SENTENCE_END = (".", "!", "?", "…")
_CLAUSE_END = (",", ";", ":", "—")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


def words_to_cues(
    words: Iterable[Word],
    *,
    max_chars: int = 42,
    max_duration: float = 3.5,
    max_gap: float = 0.8,
    min_duration: float = 0.7,
) -> list[Cue]:
    """Group recognised words into short, Shorts-style subtitle cues.

    A cue ends at a sentence boundary, a pause, or when it would get too long; a comma
    also ends it once it's at least half full, so lines break at natural points.
    """
    groups: list[list[Word]] = []
    current: list[Word] = []
    length = 0
    for word in words:
        text = word.text.strip()
        if not text:
            continue
        word = Word(word.start, word.end, text)
        if current:
            prev = current[-1]
            if (
                prev.text.endswith(_SENTENCE_END)
                or (prev.text.endswith(_CLAUSE_END) and length >= max_chars // 2)
                or length + 1 + len(text) > max_chars
                or word.end - current[0].start > max_duration
                or word.start - prev.end > max_gap
            ):
                groups.append(current)
                current, length = [], 0
        current.append(word)
        length += len(text) + (1 if length else 0)
    if current:
        groups.append(current)

    # A tail like "да?" squeezed between two cues can't be held on screen long enough to
    # read; glue it onto the previous cue instead (libass wraps it to a second line).
    merged: list[list[Word]] = []
    for group in groups:
        if merged:
            prev = merged[-1]
            too_short = group[-1].end - group[0].start < min_duration
            if (
                too_short
                and group[0].start - prev[-1].end <= max_gap
                and not prev[-1].text.endswith(_SENTENCE_END)
                and sum(len(w.text) + 1 for w in prev + group) <= max_chars * 1.5
            ):
                prev.extend(group)
                continue
        merged.append(group)
    groups = merged

    cues = []
    for i, group in enumerate(groups):
        start = group[0].start
        next_start = groups[i + 1][0].start if i + 1 < len(groups) else math.inf
        # Very short cues flash by; hold them a bit longer if the next one allows it.
        end = max(group[-1].end, min(start + min_duration, next_start))
        if end <= start:
            end = start + 0.05
        cues.append(Cue(i + 1, start, end, " ".join(w.text for w in group)))
    return cues


def extract_audio(
    video: Path,
    output: Path,
    *,
    stream: int = 0,
    start: float | None = None,
    end: float | None = None,
) -> Path:
    """Extract one audio track as 16 kHz mono WAV — the format Whisper works with."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video)]
    if end is not None:
        cmd += ["-t", f"{end - (start or 0):.3f}"]
    cmd += ["-map", f"0:a:{stream}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    cmd.append(str(output))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"Could not extract audio from {video}")
    return output


def read_wav(path: Path):
    """Load 16-bit mono WAV as float32 samples in [-1, 1].

    Passing samples directly keeps faster-whisper from decoding the file with PyAV,
    whose API changes between releases break it.
    """
    import numpy as np

    with wave.open(str(path), "rb") as f:
        frames = f.readframes(f.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(
    video: Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    audio_stream: int = 0,
    start: float | None = None,
    end: float | None = None,
    device: str = "auto",
    compute_type: str = "int8",
    max_chars: int = 42,
    on_progress: Callable[[float, float], None] | None = None,
) -> list[Cue]:
    """Recognise speech in `video` and return subtitle cues in the video's timeline."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "Speech recognition needs faster-whisper: pip install 'clipsmith[transcribe]'"
        ) from None

    offset = start or 0.0
    words: list[Word] = []
    with tempfile.TemporaryDirectory(prefix="clipsmith-") as tmp:
        wav = extract_audio(
            video, Path(tmp) / "audio.wav", stream=audio_stream, start=start, end=end
        )
        whisper = WhisperModel(model, device=device, compute_type=compute_type)
        segments, info = whisper.transcribe(
            read_wav(wav),
            language=language,
            beam_size=5,
            word_timestamps=True,
            # Skip music/silence: fewer hallucinated lines and much faster on films.
            vad_filter=True,
            # Stops Whisper from looping on a phrase it has already produced.
            condition_on_previous_text=False,
        )
        for segment in segments:  # lazy generator: recognition happens while iterating
            for w in segment.words or []:
                words.append(Word(w.start + offset, w.end + offset, w.word))
            if on_progress:
                on_progress(segment.end, info.duration)
    return words_to_cues(words, max_chars=max_chars)
