"""Ask Claude to pick the best clip-worthy moments from a subtitle transcript."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .claude import DEFAULT_MODEL, call_tool, make_client
from .metadata import META_PROPERTIES, META_REQUIRED, SHORTS_GUIDE, ShortsMeta, meta_from_dict
from .subtitles import Cue, format_timestamp

MUSIC_MARK = "♪"
DEFAULT_MAX_MUSIC = 0.15  # share of a clip that may still have original music in it


SYSTEM_PROMPT = """\
You are an experienced film editor who cuts short, shareable clips from movies.
You receive a movie transcript as numbered subtitle cues with timestamps.
Pick self-contained moments that work without extra context: a strong opening line,
a clear payoff (a joke, a reveal, an emotional beat, a memorable quote), and a clean ending.
Never start or end a clip in the middle of an exchange. Prefer moments that do not overlap.
Every clip will be published as a YouTube Shorts video, so for each one also write its
title, alternative titles, description and hashtags.

""" + SHORTS_GUIDE + """

Always answer by calling the `submit_clips` tool."""

CLIPS_TOOL: dict[str, Any] = {
    "name": "submit_clips",
    "description": "Submit the selected clips, best first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short catchy title for the clip.",
                        },
                        "start_cue": {
                            "type": "integer",
                            "description": "Number of the first cue in the clip.",
                        },
                        "end_cue": {
                            "type": "integer",
                            "description": "Number of the last cue in the clip (inclusive).",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One sentence on why this moment works as a clip.",
                        },
                        "score": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "How strong the clip is.",
                        },
                        **META_PROPERTIES,
                    },
                    "required": ["title", "start_cue", "end_cue", "reason", "score"]
                    + META_REQUIRED,
                },
            }
        },
        "required": ["clips"],
    },
}


@dataclass
class Clip:
    title: str
    start: float
    end: float
    start_cue: int
    end_cue: int
    reason: str = ""
    score: int = 0
    music: float = 0.0  # share of the clip with music in the original soundtrack
    # YouTube Shorts metadata; kept flat so the plan JSON is easy to edit by hand.
    shorts_title: str = ""
    alt_titles: list[str] = field(default_factory=list)
    description: str = ""
    hashtags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def meta(self) -> ShortsMeta:
        return ShortsMeta(self.shorts_title, list(self.alt_titles), self.description,
                          list(self.hashtags))

    def set_meta(self, meta: ShortsMeta) -> None:
        self.shorts_title = meta.title
        self.alt_titles = list(meta.alt_titles)
        self.description = meta.description
        self.hashtags = list(meta.hashtags)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Clip:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


def format_transcript(cues: list[Cue], music_map: Any = None) -> str:
    """Numbered transcript; with a music map, cues over original music get a ♪ mark."""
    lines = []
    for c in cues:
        text = c.text.replace("\n", " / ")
        mark = ""
        if music_map is not None and music_map.fraction(c.start, c.end) >= 0.5:
            mark = f"{MUSIC_MARK} "
        lines.append(f"[{c.index}] {format_timestamp(c.start, '.')[:-4]} {mark}{text}")
    return "\n".join(lines)


def build_request(
    cues: list[Cue],
    *,
    count: int,
    min_duration: float,
    max_duration: float,
    theme: str | None,
    language: str | None,
    model: str,
    movie: str | None = None,
    music_map: Any = None,
) -> dict[str, Any]:
    task = [
        f"Select {count} clips (fewer only if there truly aren't enough good moments).",
        f"Each clip must last between {min_duration:.0f} and {max_duration:.0f} seconds "
        "(measured from the start of the first cue to the end of the last cue).",
    ]
    if theme:
        task.append(f"Focus on this theme / request from the user: {theme}")
    if music_map is not None:
        task.append(
            f"Cues marked {MUSIC_MARK} play over music from the original soundtrack. The clips "
            "get our own copyright-free music instead, so choose moments WITHOUT "
            f"{MUSIC_MARK} cues; a clip with them will be rejected."
        )
        windows = music_free_windows(music_map, min_duration, cues[-1].end)
        if windows:
            listed = ", ".join(
                f"{format_timestamp(a, '.')[:-4]}-{format_timestamp(b, '.')[:-4]}"
                for a, b in windows
            )
            task.append(f"Every clip must fit inside one of these music-free windows: {listed}.")
    task.append(
        f"Movie / series: {movie}." if movie else "The movie name is unknown - do not guess it."
    )
    if language:
        task.append(f"Write all texts in {language}.")
    else:
        task.append("Write all texts in the language of the transcript.")

    return {
        "model": model,
        "max_tokens": 8192,
        "system": SYSTEM_PROMPT,
        "tools": [CLIPS_TOOL],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"<transcript>\n{format_transcript(cues, music_map)}\n</transcript>"
                        ),
                        # The transcript is the bulk of the prompt; cache it so re-runs
                        # with a different theme or count are cheaper.
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "\n".join(task)},
                ],
            }
        ],
    }


def clips_from_response(
    raw_clips: list[dict[str, Any]],
    cues: list[Cue],
    *,
    max_duration: float,
) -> list[Clip]:
    """Turn cue ranges from Claude into timed clips, dropping invalid ones."""
    by_index = {c.index: c for c in cues}
    clips: list[Clip] = []
    for item in raw_clips:
        try:
            a, b = int(item["start_cue"]), int(item["end_cue"])
        except (KeyError, TypeError, ValueError):
            continue
        if a > b:
            a, b = b, a
        if a not in by_index or b not in by_index:
            continue
        start, end = by_index[a].start, by_index[b].end
        # Allow some slack: the model is told the limits but can't measure precisely.
        if end - start > max_duration * 1.5:
            continue
        clip = Clip(
            title=str(item.get("title", "")).strip() or f"Clip {len(clips) + 1}",
            start=start,
            end=end,
            start_cue=a,
            end_cue=b,
            reason=str(item.get("reason", "")).strip(),
            score=int(item.get("score", 0) or 0),
        )
        clip.set_meta(meta_from_dict(item))
        clips.append(clip)
    clips.sort(key=lambda c: c.score, reverse=True)
    return clips


def music_free_windows(
    music_map: Any, min_duration: float, total: float
) -> list[tuple[float, float]]:
    """Stretches without original music that are long enough to hold a clip."""
    windows, prev = [], 0.0
    for a, b in music_map.segments() + [(total, total)]:
        if a - prev >= min_duration:
            windows.append((prev, a))
        prev = max(prev, b)
    return windows


def drop_music_clips(
    clips: list[Clip], music_map: Any, max_music: float = DEFAULT_MAX_MUSIC
) -> tuple[list[Clip], list[Clip]]:
    """Split clips into (kept, rejected) by how much original music they contain."""
    kept, rejected = [], []
    for clip in clips:
        clip.music = round(music_map.fraction(clip.start, clip.end), 2)
        (kept if clip.music <= max_music else rejected).append(clip)
    return kept, rejected


def suggest_clips(
    cues: list[Cue],
    *,
    count: int = 5,
    min_duration: float = 20,
    max_duration: float = 90,
    theme: str | None = None,
    language: str | None = None,
    movie: str | None = None,
    music_map: Any = None,
    max_music: float = DEFAULT_MAX_MUSIC,
    model: str = DEFAULT_MODEL,
    client: Any = None,
) -> list[Clip]:
    """Ask Claude for clips. With `music_map`, clips over original music are dropped."""
    if not cues:
        raise ValueError("Subtitles are empty — nothing to analyze.")
    if client is None:
        client = make_client()

    request = build_request(
        cues,
        # Ask for spares: some may still be rejected for music.
        count=count * 2 if music_map is not None else count,
        min_duration=min_duration,
        max_duration=max_duration,
        theme=theme,
        language=language,
        model=model,
        movie=movie,
        music_map=music_map,
    )
    result = call_tool(client, request, "submit_clips")
    clips = clips_from_response(result.get("clips", []), cues, max_duration=max_duration)
    if music_map is not None:
        clips, _ = drop_music_clips(clips, music_map, max_music)
    return clips[:count]
