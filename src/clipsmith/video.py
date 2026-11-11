"""Cutting clips and rendering subtitles with ffmpeg."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .subtitles import Cue, slice_cues, to_srt

DEFAULT_STYLE = "FontName=Arial,FontSize=20,Outline=2,Shadow=0,MarginV=24"
# libass sizes are relative to frame height, so a 9:16 frame needs a smaller font to fit
# a line across. The higher margin keeps text above the Shorts UI (title, buttons).
VERTICAL_STYLE = "FontName=Arial,FontSize=11,Bold=1,Outline=1.5,Shadow=0,MarginV=55,MarginL=8,MarginR=8"

# "blur" layout on a 1080x1920 canvas: a 3:4 picture (1080x1440) shifted up over a
# blurred copy of the frame. The tall bottom band sits under the Shorts title/description
# overlay, so nothing important is hidden there.
PANEL_TOP = 120
PANEL_BOTTOM = 1920 - 1440 - PANEL_TOP
# Keeps subtitles inside the picture, just above the bottom band (libass units are
# 1/288 of frame height, so 1 unit is ~6.7 px here).
PANEL_STYLE = "FontName=Arial,FontSize=11,Bold=1,Outline=1.5,Shadow=0,MarginV=62,MarginL=8,MarginR=8"

_BLURRED_BACKGROUND = (
    "[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
    "boxblur=20:2,eq=brightness=-0.08[bgb];"
)

LAYOUTS = ("blur", "fit", "crop", "original")
DEFAULT_LAYOUT = "blur"
_LAYOUT_STYLES = {"blur": PANEL_STYLE, "original": DEFAULT_STYLE}
_LAYOUT_FILTERS = {
    # 3:4 centre crop, shifted up, over a blurred background.
    "blur": (
        "split=2[bg][fg];"
        + _BLURRED_BACKGROUND
        + "[fg]crop=ih*3/4:ih,scale=1080:1440[fgs];"
        f"[bgb][fgs]overlay=0:{PANEL_TOP},setsar=1"
    ),
    # Centre crop to 9:16, then upscale to Shorts resolution.
    "crop": "crop=ih*9/16:ih,scale=1080:1920,setsar=1",
    # Whole frame, letterboxed over a blurred, zoomed copy of itself.
    "fit": (
        "split=2[bg][fg];"
        + _BLURRED_BACKGROUND
        + "[fg]scale=1080:-2[fgs];"
        "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1"
    ),
}


# Loudness targets. Film mixes are quiet (often -25 LUFS); Shorts play around -14, so the
# dialogue is brought up to DIALOGUE_LUFS and a limiter catches the peaks.
DIALOGUE_LUFS = -16
# Music level relative to the dialogue before ducking; under speech it drops ~6 dB more.
DEFAULT_MUSIC_LEVEL = -10.0
_FMT = "aformat=sample_rates=48000:channel_layouts=stereo"
# loudnorm resamples to 192 kHz internally, hence the aformat after it.
_DIALOGUE = f"loudnorm=I={DIALOGUE_LUFS}:TP=-1.5:LRA=11,{_FMT}"
_LIMITER = "alimiter=limit=0.89:level=disabled"


class FFmpegError(RuntimeError):
    pass


def music_filter(duration: float, level_db: float = DEFAULT_MUSIC_LEVEL) -> str:
    """Audio graph mixing a background track (input 1) under the original sound (input 0).

    Both are loudness-normalised first, so the result doesn't depend on how the track was
    mastered. The track fades in/out and is ducked by the dialogue (sidechain), so lines
    stay intelligible and the music swells in the pauses.
    """
    fade_out = max(0.0, duration - 2.0)
    music_lufs = DIALOGUE_LUFS + level_db
    return (
        f"[1:a]loudnorm=I={music_lufs:.1f}:TP=-3,{_FMT},afade=t=in:d=1,"
        f"afade=t=out:st={fade_out:.3f}:d=2[mus];"
        f"[0:a]{_DIALOGUE},asplit=2[orig][key];"
        "[mus][key]sidechaincompress=threshold=0.05:ratio=4:attack=50:release=600[duck];"
        f"[orig][duck]amix=inputs=2:duration=first:normalize=0,{_LIMITER}[aout]"
    )


def ensure_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(f"Not found in PATH: {', '.join(missing)}. Please install ffmpeg.")


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"ffprobe failed on {video}")
    return float(json.loads(result.stdout)["format"]["duration"])


def subtitle_stream_count(video: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or f"ffprobe failed on {video}")
    return len(result.stdout.split())


def extract_subtitles(video: Path, output: Path, stream: int = 0) -> Path:
    """Extract an embedded text subtitle track (e.g. from .mkv) to an .srt file."""
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video), "-map", f"0:s:{stream}", "-c:s", "srt", str(output),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(
            f"Could not extract subtitle track {stream} from {video.name}. "
            "Image-based subtitles (PGS/VobSub) are not supported — pass an .srt with --subs.\n"
            + result.stderr.strip()
        )
    return output


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len].strip("-") or "clip"


def build_command(
    video: Path,
    output: Path,
    *,
    start: float,
    duration: float,
    srt_name: str | None,
    soft_subs_path: Path | None = None,
    layout: str = DEFAULT_LAYOUT,
    style: str | None = None,
    crf: int = 20,
    music: Path | None = None,
    music_level: float = DEFAULT_MUSIC_LEVEL,
) -> list[str]:
    """Build the ffmpeg command line.

    `srt_name` is a bare file name relative to ffmpeg's working directory: the `subtitles`
    filter has painful escaping rules for full paths, so we avoid them entirely.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    # -ss before -i is fast, and accurate because we re-encode. Timestamps restart at 0,
    # which matches the shifted subtitle file.
    cmd += ["-ss", f"{start:.3f}", "-i", str(video.resolve())]
    if music is not None:
        # Looped, so short tracks still cover the whole clip; -t below trims it.
        cmd += ["-stream_loop", "-1", "-i", str(music.resolve())]
    if soft_subs_path is not None:
        cmd += ["-i", str(soft_subs_path)]
    subs_input = 2 if music is not None else 1

    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout} (expected one of {', '.join(LAYOUTS)})")
    if style is None:
        style = _LAYOUT_STYLES.get(layout, VERTICAL_STYLE)

    filters = []
    if layout in _LAYOUT_FILTERS:
        filters.append(_LAYOUT_FILTERS[layout])
    if srt_name:
        filters.append(f"subtitles={srt_name}:force_style='{style}'")

    if music is not None:
        video_graph = f"[0:v]{','.join(filters) or 'null'}[vout]"
        cmd += ["-filter_complex", f"{video_graph};{music_filter(duration, music_level)}"]
        cmd += ["-t", f"{duration:.3f}", "-map", "[vout]", "-map", "[aout]"]
    else:
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-af", f"{_DIALOGUE},{_LIMITER}"]
        cmd += ["-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "0:a:0?"]
    if soft_subs_path is not None:
        cmd += ["-map", f"{subs_input}:s:0", "-c:s", "mov_text"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(output.resolve()),
    ]
    return cmd


def render_clip(
    video: Path,
    cues: list[Cue],
    output: Path,
    *,
    start: float,
    end: float,
    padding: float = 0.5,
    subs: str = "burn",
    layout: str = DEFAULT_LAYOUT,
    style: str | None = None,
    video_duration: float | None = None,
    music: Path | None = None,
    music_level: float = DEFAULT_MUSIC_LEVEL,
) -> Path:
    """Render one clip.

    `subs` is "burn" (hard subtitles), "soft" (selectable track) or "none".
    `layout` is "blur" / "fit" / "crop" (vertical 1080x1920 for Shorts) or "original".
    `music` is an optional background track mixed under the original sound.
    """
    if subs not in ("burn", "soft", "none"):
        raise ValueError(f"Unknown subtitle mode: {subs}")

    start = max(0.0, start - padding)
    end = end + padding
    if video_duration is not None:
        end = min(end, video_duration)
    if end <= start:
        raise ValueError(f"Empty clip: {start:.2f}–{end:.2f}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clipsmith-") as tmp:
        tmp_dir = Path(tmp)
        srt_path = tmp_dir / "clip.srt"
        clip_cues = slice_cues(cues, start, end)
        if not clip_cues:
            subs = "none"
        srt_path.write_text(to_srt(clip_cues), encoding="utf-8")

        cmd = build_command(
            video,
            output,
            start=start,
            duration=end - start,
            srt_name=srt_path.name if subs == "burn" else None,
            soft_subs_path=srt_path if subs == "soft" else None,
            layout=layout,
            style=style,
            music=music,
            music_level=music_level,
        )
        result = subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegError(result.stderr.strip() or "ffmpeg failed")

        # Ship a matching .srt next to the clip — handy for uploading to platforms.
        if clip_cues:
            shutil.copyfile(srt_path, output.with_suffix(".srt"))
    return output
