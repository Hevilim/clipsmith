import shutil
import subprocess
from pathlib import Path

import pytest

from clipsmith import video
from clipsmith.subtitles import Cue

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_slugify():
    assert video.slugify("Hello, World!  Part 2") == "hello-world-part-2"
    assert video.slugify("Привет мир") == "привет-мир"
    assert video.slugify("???") == "clip"


def test_build_command_burn_vertical():
    cmd = video.build_command(
        Path("movie.mkv"), Path("out.mp4"),
        start=12.5, duration=30, srt_name="clip.srt", layout="crop",
    )
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("crop=") and "subtitles=clip.srt" in vf
    assert video.VERTICAL_STYLE in vf
    assert cmd[cmd.index("-ss") + 1] == "12.500"
    assert cmd[cmd.index("-t") + 1] == "30.000"


def test_build_command_fit_layout():
    cmd = video.build_command(
        Path("movie.mkv"), Path("out.mp4"),
        start=0, duration=5, srt_name="clip.srt", layout="fit",
    )
    vf = cmd[cmd.index("-vf") + 1]
    assert "boxblur" in vf and "overlay=(W-w)/2:(H-h)/2" in vf
    assert vf.index("overlay") < vf.index("subtitles=")


def test_build_command_blur_panel_is_default():
    cmd = video.build_command(
        Path("movie.mkv"), Path("out.mp4"), start=0, duration=5, srt_name="clip.srt",
    )
    vf = cmd[cmd.index("-vf") + 1]
    assert "crop=ih*3/4:ih" in vf and "boxblur" in vf
    assert f"overlay=0:{video.PANEL_TOP}" in vf
    assert vf.index("overlay") < vf.index("subtitles=")
    assert video.PANEL_STYLE in vf
    assert video.PANEL_TOP + 1440 + video.PANEL_BOTTOM == 1920


def test_build_command_rejects_unknown_layout():
    with pytest.raises(ValueError):
        video.build_command(
            Path("m.mkv"), Path("o.mp4"), start=0, duration=1, srt_name=None, layout="square"
        )


def test_build_command_soft_subs():
    cmd = video.build_command(
        Path("movie.mkv"), Path("out.mp4"),
        start=0, duration=5, srt_name=None, soft_subs_path=Path("/tmp/x.srt"),
        layout="original",
    )
    assert "-vf" not in cmd
    assert "mov_text" in cmd and "1:s:0" in cmd


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize(
    "mode,layout",
    [
        ("burn", "blur"), ("burn", "fit"), ("burn", "crop"),
        ("soft", "original"), ("none", "original"),
    ],
)
def test_render_clip_end_to_end(tmp_path, mode, layout):
    src = tmp_path / "movie.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
            "-f", "lavfi", "-i", "sine=duration=6",
            "-c:v", "libx264", "-c:a", "aac", "-shortest", str(src),
        ],
        check=True,
    )
    cues = [Cue(1, 1.0, 2.5, "Hello"), Cue(2, 3.0, 4.0, "World")]
    out = tmp_path / "out" / "clip.mp4"
    video.render_clip(
        src, cues, out, start=1.0, end=4.0, padding=0.5, subs=mode, layout=layout,
        video_duration=video.probe_duration(src),
    )
    assert out.exists()
    if layout != "original":
        size = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert size == "1080,1920"
    assert 3.0 <= video.probe_duration(out) <= 4.3
    assert "Hello" in out.with_suffix(".srt").read_text()
