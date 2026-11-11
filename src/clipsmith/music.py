"""Find where the original soundtrack has music, so clips can avoid it.

Uses the Audio Spectrogram Transformer fine-tuned on AudioSet (ONNX export, ~90 MB,
CPU-only via onnxruntime). Every few seconds of audio get a "music" probability.
"""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

MODEL_REPO = "Xenova/ast-finetuned-audioset-10-10-0.4593"
MODEL_FILE = "onnx/model_quantized.onnx"

SAMPLE_RATE = 16000
WINDOW = 10.24  # seconds of audio the model looks at (1024 frames x 10 ms)
# Windows overlap by WINDOW - hop; 5 s is plenty for picking clips and scans a 45-minute
# episode in ~6 minutes on an 8-core CPU.
DEFAULT_HOP = 5.0
DEFAULT_THRESHOLD = 0.3
# AudioSet classes that mean "there is music": Music, Musical instrument, Singing, Song,
# Background music, Theme music, Soundtrack music.
MUSIC_CLASSES = (137, 138, 27, 266, 267, 268, 270)

# ASTFeatureExtractor settings from the model's preprocessor_config.json.
_NUM_MEL_BINS = 128
_MAX_FRAMES = 1024
_MEAN, _STD = -4.2677393, 4.5689974


# --- results ------------------------------------------------------------------


@dataclass
class MusicMap:
    """Music probability over time: one score per `hop` seconds, starting at 0."""

    hop: float
    scores: list[float]
    model: str = MODEL_REPO
    threshold: float = DEFAULT_THRESHOLD
    meta: dict = field(default_factory=dict)

    def score_at(self, t: float) -> float:
        if not self.scores:
            return 0.0
        i = min(max(int(t / self.hop), 0), len(self.scores) - 1)
        return self.scores[i]

    def fraction(self, start: float, end: float, threshold: float | None = None) -> float:
        """Share of [start, end] where music probability is above the threshold."""
        threshold = self.threshold if threshold is None else threshold
        if end <= start or not self.scores:
            return 0.0
        first, last = int(start / self.hop), int(math.ceil(end / self.hop))
        covered = 0.0
        for i in range(first, last):
            lo, hi = max(start, i * self.hop), min(end, (i + 1) * self.hop)
            if hi > lo and self.score_at(i * self.hop) >= threshold:
                covered += hi - lo
        return covered / (end - start)

    def segments(self, threshold: float | None = None, min_gap: float = 5.0) -> list[tuple[float, float]]:
        """Merged [start, end] intervals with music."""
        threshold = self.threshold if threshold is None else threshold
        out: list[list[float]] = []
        for i, s in enumerate(self.scores):
            if s < threshold:
                continue
            a, b = i * self.hop, (i + 1) * self.hop
            if out and a - out[-1][1] <= min_gap:
                out[-1][1] = b
            else:
                out.append([a, b])
        return [(a, b) for a, b in out]

    def save(self, path: Path) -> None:
        data = {"model": self.model, "hop": self.hop, "threshold": self.threshold,
                "meta": self.meta, "scores": [round(s, 3) for s in self.scores]}
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MusicMap:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(hop=data["hop"], scores=data["scores"], model=data.get("model", MODEL_REPO),
                   threshold=data.get("threshold", DEFAULT_THRESHOLD), meta=data.get("meta", {}))


def map_path(video: Path) -> Path:
    return video.with_suffix(".music.json")


# --- background tracks ------------------------------------------------------------

AUDIO_EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus")


def list_tracks(path: Path) -> list[Path]:
    """A single audio file, or every audio file in a folder (sorted)."""
    if path.is_dir():
        tracks = sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if not tracks:
            raise ValueError(f"No audio files in {path}")
        return tracks
    if not path.exists():
        raise FileNotFoundError(f"Music not found: {path}")
    return [path]


def track_credit(track: Path) -> str:
    """Attribution text from `<track>.txt`, if present (e.g. for CC-BY music)."""
    credit = track.with_suffix(".txt")
    return credit.read_text(encoding="utf-8").strip() if credit.is_file() else ""


# --- features -------------------------------------------------------------------


def _mel_banks(n_fft: int = 512, low: float = 20.0, high: float = SAMPLE_RATE / 2):
    """Kaldi-style triangular mel filters, as torchaudio.compliance.kaldi builds them."""
    import numpy as np

    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    n_bins = n_fft // 2
    fft_mels = mel(np.arange(n_bins) * SAMPLE_RATE / n_fft)
    mel_low, mel_high = mel(low), mel(high)
    delta = (mel_high - mel_low) / (_NUM_MEL_BINS + 1)
    left = mel_low + np.arange(_NUM_MEL_BINS)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (fft_mels - left) / (center - left)
    down = (right - fft_mels) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)  # zero weight for Nyquist bin


def fbank(samples, banks=None):
    """Kaldi fbank features (128 log-mel bins, 25 ms / 10 ms), as the AST model expects.

    `samples` are float32 in [-1, 1] at 16 kHz.
    """
    import numpy as np

    banks = _mel_banks() if banks is None else banks
    x = np.asarray(samples, dtype=np.float32) * 32768.0
    frame_len, shift, n_fft = 400, 160, 512
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    n_frames = 1 + (len(x) - frame_len) // shift
    idx = np.arange(frame_len)[None, :] + shift * np.arange(n_frames)[:, None]
    frames = x[idx].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    prev = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * prev
    frames *= np.hanning(frame_len)  # symmetric Hann == Kaldi's "hanning"
    power = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel_energy = power.astype(np.float32) @ banks.T
    return np.log(np.maximum(mel_energy, np.finfo(np.float32).eps))


def model_input(samples, banks=None):
    import numpy as np

    feats = fbank(samples, banks)[:_MAX_FRAMES]
    if len(feats) < _MAX_FRAMES:
        feats = np.pad(feats, ((0, _MAX_FRAMES - len(feats)), (0, 0)))
    return ((feats - _MEAN) / (2 * _STD)).astype(np.float32)


# --- scanning -------------------------------------------------------------------


def _load_session():
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "Music detection needs onnxruntime: pip install 'clipsmith[music]'"
        ) from None
    path = hf_hub_download(MODEL_REPO, MODEL_FILE)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def music_probability(logits) -> list[float]:
    import numpy as np

    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    return probs[:, list(MUSIC_CLASSES)].max(axis=1).tolist()


def scan(
    video: Path,
    *,
    audio_stream: int = 0,
    hop: float = DEFAULT_HOP,
    batch: int = 8,
    on_progress: Callable[[float, float], None] | None = None,
) -> MusicMap:
    """Score every `hop` seconds of the video's audio for music."""
    from .transcribe import extract_audio, read_wav

    with tempfile.TemporaryDirectory(prefix="clipsmith-") as tmp:
        audio = read_wav(extract_audio(video, Path(tmp) / "audio.wav", stream=audio_stream))
    session = _load_session()
    input_name = session.get_inputs()[0].name
    banks = _mel_banks()

    import numpy as np

    total = len(audio) / SAMPLE_RATE
    n = max(1, int(math.ceil(total / hop)))
    win, half = int(WINDOW * SAMPLE_RATE), int((WINDOW - hop) / 2 * SAMPLE_RATE)
    scores: list[float] = []
    for first in range(0, n, batch):
        inputs = []
        for i in range(first, min(first + batch, n)):
            # Centre the 10 s window on this hop so the score describes it.
            a = max(0, int(i * hop * SAMPLE_RATE) - half)
            inputs.append(model_input(audio[a:a + win], banks))
        logits = session.run(None, {input_name: np.stack(inputs)})[0]
        scores.extend(music_probability(logits))
        if on_progress:
            on_progress(min(len(scores) * hop, total), total)
    return MusicMap(hop=hop, scores=scores, meta={"audio_stream": audio_stream})


def load_or_scan(
    video: Path, *, audio_stream: int = 0, on_progress=None, log=print
) -> MusicMap:
    path = map_path(video)
    if path.exists():
        cached = MusicMap.load(path)
        if cached.meta.get("audio_stream", 0) == audio_stream:
            return cached
    log(f"Scanning {video.name} for music (one-time, cached in {path.name})…")
    result = scan(video, audio_stream=audio_stream, on_progress=on_progress)
    result.save(path)
    return result
