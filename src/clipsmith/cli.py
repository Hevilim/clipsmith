"""Command-line interface for clipsmith."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__, subtitles, video
from .analyze import DEFAULT_MODEL, Clip, suggest_clips
from .subtitles import Cue, format_timestamp

SUB_EXTENSIONS = (".srt", ".vtt")


def parse_time(value: str) -> float:
    """Accept seconds ("754.5") or timestamps ("12:34.5", "01:12:34,500")."""
    value = value.strip().replace(",", ".")
    try:
        seconds = 0.0
        for part in value.split(":"):
            seconds = seconds * 60 + float(part)
        return seconds
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid time: {value!r}") from None


def short_ts(seconds: float) -> str:
    return format_timestamp(seconds, ".")[:-4]


def load_cues(
    video_path: Path, subs: Path | None, stream: int, *, required: bool = True
) -> list[Cue]:
    """Find and load subtitles. With required=False, returns [] if there are none."""
    if subs is None:
        for ext in SUB_EXTENSIONS:
            candidate = video_path.with_suffix(ext)
            if candidate.exists():
                subs = candidate
                break
    if subs is not None:
        print(f"Subtitles: {subs}", file=sys.stderr)
        return subtitles.load(subs)

    if video.subtitle_stream_count(video_path) == 0:
        if required:
            raise ValueError(
                f"No subtitles found for {video_path.name}: pass --subs, put "
                f"{video_path.with_suffix('.srt').name} next to the video, or generate it "
                f"with `clipsmith transcribe {video_path}`."
            )
        print("Subtitles: none found, rendering without them", file=sys.stderr)
        return []

    print(f"Subtitles: embedded track #{stream} from {video_path.name}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="clipsmith-") as tmp:
        srt = video.extract_subtitles(video_path, Path(tmp) / "embedded.srt", stream)
        return subtitles.load(srt)


def print_clips(clips: list[Clip]) -> None:
    if not clips:
        print("No clips found.")
        return
    for n, clip in enumerate(clips, start=1):
        print(
            f"{n:>2}. [{clip.score}/10] {clip.title}\n"
            f"    {short_ts(clip.start)} → {short_ts(clip.end)}  ({clip.duration:.0f}s)\n"
            f"    {clip.reason}"
        )


def save_plan(path: Path, video_path: Path, clips: list[Clip]) -> None:
    data = {"video": str(video_path), "clips": [c.to_dict() for c in clips]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPlan saved to {path} — edit it if needed, then run `clipsmith render`.")


def load_plan(path: Path) -> list[Clip]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Clip.from_dict(c) for c in data["clips"]]


def parse_selection(value: str | None, total: int) -> list[int]:
    if not value:
        return list(range(total))
    picked = []
    for part in value.split(","):
        n = int(part)
        if not 1 <= n <= total:
            raise SystemExit(f"--only: clip {n} is out of range 1..{total}")
        picked.append(n - 1)
    return picked


def render_all(args: argparse.Namespace, cues: list[Cue], clips: list[Clip]) -> None:
    video.ensure_ffmpeg()
    duration = video.probe_duration(args.video)
    out_dir: Path = args.out
    subs_mode = args.subs_mode if cues else "none"
    for n, clip in enumerate(clips, start=1):
        name = f"{n:02d}-{video.slugify(clip.title)}.mp4"
        target = out_dir / name
        print(f"[{n}/{len(clips)}] {clip.title} → {target}", file=sys.stderr)
        video.render_clip(
            args.video,
            cues,
            target,
            start=clip.start,
            end=clip.end,
            padding=args.padding,
            subs=subs_mode,
            layout=args.layout,
            video_duration=duration,
        )
    print(f"Done: {len(clips)} clip(s) in {out_dir}", file=sys.stderr)


# --- commands ---------------------------------------------------------------


def cmd_suggest(args: argparse.Namespace) -> None:
    cues = load_cues(args.video, args.subs, args.subs_stream)
    clips = _suggest(args, cues)
    print_clips(clips)
    if clips:
        save_plan(args.plan or args.video.with_suffix(".plan.json"), args.video, clips)


def cmd_render(args: argparse.Namespace) -> None:
    cues = load_cues(args.video, args.subs, args.subs_stream, required=False)
    clips = load_plan(args.plan)
    clips = [clips[i] for i in parse_selection(args.only, len(clips))]
    render_all(args, cues, clips)


def cmd_run(args: argparse.Namespace) -> None:
    cues = load_cues(args.video, args.subs, args.subs_stream)
    clips = _suggest(args, cues)
    print_clips(clips)
    if clips:
        render_all(args, cues, clips)


def cmd_cut(args: argparse.Namespace) -> None:
    cues = load_cues(args.video, args.subs, args.subs_stream, required=False)
    if args.end <= args.start:
        raise SystemExit("--end must be after --start")
    clip = Clip(title=args.title, start=args.start, end=args.end, start_cue=0, end_cue=0)
    render_all(args, cues, [clip])


def cmd_transcribe(args: argparse.Namespace) -> None:
    from . import transcribe

    output: Path = args.output or args.video.with_suffix(".srt")
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists (use --force to overwrite)")

    def progress(done: float, total: float) -> None:
        pct = 100 * done / total if total else 0
        print(f"\r  {short_ts(done)} / {short_ts(total)}  ({pct:.0f}%)", end="", file=sys.stderr)

    print(
        f"Transcribing audio track #{args.audio_stream} with {args.model} "
        "(the model is downloaded on first use)…",
        file=sys.stderr,
    )
    cues = transcribe.transcribe(
        args.video,
        model=args.model,
        language=args.language,
        audio_stream=args.audio_stream,
        start=args.start,
        end=args.end,
        device=args.device,
        max_chars=args.max_chars,
        on_progress=progress,
    )
    print(file=sys.stderr)
    if not cues:
        raise SystemExit("No speech recognised.")
    output.write_text(subtitles.to_srt(cues), encoding="utf-8")
    print(f"Saved {len(cues)} subtitles to {output}", file=sys.stderr)


def _suggest(args: argparse.Namespace, cues: list[Cue]) -> list[Clip]:
    print(f"Asking {args.model} to pick clips from {len(cues)} cues…", file=sys.stderr)
    return suggest_clips(
        cues,
        count=args.count,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        theme=args.theme,
        language=args.language,
        model=args.model,
    )


# --- argument parsing ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipsmith",
        description="Cut subtitled clips from movies, with Claude picking the best moments.",
    )
    parser.add_argument("--version", action="version", version=f"clipsmith {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("video", type=Path, help="Path to the movie file")
    common.add_argument(
        "--subs", type=Path, help="Subtitle file (.srt/.vtt). Default: next to the video, "
        "or the embedded track",
    )
    common.add_argument(
        "--subs-stream", type=int, default=0, help="Embedded subtitle track index (default: 0)"
    )

    ai = argparse.ArgumentParser(add_help=False)
    ai.add_argument("-n", "--count", type=int, default=5, help="How many clips (default: 5)")
    ai.add_argument("--min-duration", type=float, default=20, help="Seconds (default: 20)")
    ai.add_argument("--max-duration", type=float, default=90, help="Seconds (default: 90)")
    ai.add_argument("--theme", help='What to look for, e.g. "funniest jokes" or "plot twists"')
    ai.add_argument("--language", help="Language for titles/reasons (default: transcript's)")
    ai.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL})")

    render = argparse.ArgumentParser(add_help=False)
    render.add_argument("-o", "--out", type=Path, default=Path("clips"), help="Output directory")
    render.add_argument(
        "--subs-mode", choices=("burn", "soft", "none"), default="burn",
        help="burn = hardcoded, soft = selectable track, none = no subtitles (default: burn)",
    )
    render.add_argument(
        "--layout", choices=video.LAYOUTS, default=video.DEFAULT_LAYOUT,
        help="blur = 3:4 picture shifted up over a blurred background; "
        "fit = whole frame over a blurred background; crop = 9:16 centre crop; "
        f"original = keep source aspect (default: {video.DEFAULT_LAYOUT})",
    )
    render.add_argument(
        "--padding", type=float, default=0.5, help="Extra seconds before/after (default: 0.5)"
    )

    p = sub.add_parser("suggest", parents=[common, ai], help="Ask Claude for clips, save a plan")
    p.add_argument("--plan", type=Path, help="Where to save the plan (default: <video>.plan.json)")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("render", parents=[common, render], help="Render clips from a plan file")
    p.add_argument("--plan", type=Path, required=True, help="Plan JSON from `suggest`")
    p.add_argument("--only", help="Comma-separated clip numbers to render, e.g. 1,3")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("run", parents=[common, ai, render], help="Suggest and render in one go")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("cut", parents=[common, render], help="Cut a clip manually, no AI")
    p.add_argument("--start", type=parse_time, required=True, help="e.g. 01:02:03.5")
    p.add_argument("--end", type=parse_time, required=True, help="e.g. 01:03:10")
    p.add_argument("--title", default="clip", help="Used for the file name")
    p.set_defaults(func=cmd_cut)

    from .transcribe import DEFAULT_MODEL as WHISPER_MODEL

    p = sub.add_parser("transcribe", help="Generate subtitles from speech (faster-whisper)")
    p.add_argument("video", type=Path, help="Path to the movie file")
    p.add_argument("-o", "--output", type=Path, help="Output .srt (default: next to the video)")
    p.add_argument("--force", action="store_true", help="Overwrite an existing .srt")
    p.add_argument(
        "--model", default=WHISPER_MODEL,
        help=f"Whisper model: tiny, base, small, medium, large-v3, ... (default: {WHISPER_MODEL})",
    )
    p.add_argument("--language", help='Speech language code, e.g. "ru" (default: auto-detect)')
    p.add_argument("--audio-stream", type=int, default=0, help="Audio track index (default: 0)")
    p.add_argument("--start", type=parse_time, help="Only transcribe from this time")
    p.add_argument("--end", type=parse_time, help="Only transcribe up to this time")
    p.add_argument("--device", default="auto", help="cpu, cuda or auto (default: auto)")
    p.add_argument(
        "--max-chars", type=int, default=42, help="Max characters per subtitle (default: 42)"
    )
    p.set_defaults(func=cmd_transcribe)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (video.FFmpegError, ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        if type(exc).__module__.startswith("anthropic"):
            raise SystemExit(f"Claude API error: {exc}") from None
        raise


if __name__ == "__main__":
    main()
