"""YouTube Shorts title / description / hashtags, written by Claude."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .claude import call_tool, make_client
from .subtitles import Cue, format_timestamp

TITLE_LIMIT = 100  # hard YouTube limit
DESCRIPTION_LIMIT = 5000  # hard YouTube limit
MAX_HASHTAGS = 5  # more than ~15 makes YouTube ignore them all; 3-5 is the sweet spot

SHORTS_GUIDE = """\
How to write YouTube Shorts metadata that gets views:
- Title: 40-60 characters (the feed truncates longer ones), hard limit 100. Put the hook in
  the first 3-4 words. Create a curiosity gap or name the emotion ("He came to the wrong
  door", "Nobody expected this answer") - tease the payoff, never spoil it. Concrete beats
  generic. At most one emoji, no ALL CAPS, no hashtags in the title.
- Never lie about what happens in the clip: misleading titles kill retention, and
  retention is what the Shorts algorithm rewards.
- Alternative titles: two titles built on a different angle (question / quote from the
  clip / emotion), for YouTube's title A/B test.
- Description: 2-4 short lines. Line 1 is a hook or a question that makes people comment
  (comments boost reach). Then one line of context without spoilers. Mention the movie or
  series name only if the user gave it - never guess it. End with a short call to action.
  No hashtags in the description text itself.
- Hashtags: 3-5, the first one is #shorts, then the movie/series (only if known), genre
  and topic. Popular, real tags in the clip's language; no spaces inside a tag.
- Write in the language of the transcript unless told otherwise; sound like a person,
  not an ad."""

META_PROPERTIES: dict[str, Any] = {
    "shorts_title": {"type": "string", "description": "YouTube Shorts title."},
    "alt_titles": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Two alternative titles for an A/B test.",
    },
    "description": {"type": "string", "description": "Description without hashtags."},
    "hashtags": {
        "type": "array",
        "items": {"type": "string"},
        "description": "3-5 hashtags, starting with #shorts.",
    },
}
META_REQUIRED = list(META_PROPERTIES)

METADATA_TOOL: dict[str, Any] = {
    "name": "submit_metadata",
    "description": "Submit the YouTube Shorts metadata for the clip.",
    "input_schema": {"type": "object", "properties": META_PROPERTIES, "required": META_REQUIRED},
}


@dataclass
class ShortsMeta:
    title: str = ""
    alt_titles: list[str] = field(default_factory=list)
    description: str = ""
    hashtags: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.title)

    def full_description(self) -> str:
        """Description as it should be pasted into YouTube, hashtags included."""
        return "\n\n".join(p for p in (self.description, " ".join(self.hashtags)) if p)

    def to_text(self) -> str:
        parts = [f"НАЗВАНИЕ\n{self.title}"]
        if self.alt_titles:
            alts = "\n".join(f"- {t}" for t in self.alt_titles)
            parts.append(f"ВАРИАНТЫ НАЗВАНИЯ (A/B-тест)\n{alts}")
        parts.append(f"ОПИСАНИЕ\n{self.full_description()}")
        return "\n\n".join(parts) + "\n"


def _clip_text(text: Any, limit: int) -> str:
    text = re.sub(r"[ \t]+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:—-") + "…"


def normalize_hashtags(tags: Any) -> list[str]:
    out: list[str] = []
    for tag in tags if isinstance(tags, list) else str(tags or "").split():
        tag = re.sub(r"[^\w]", "", str(tag), flags=re.UNICODE)
        if tag and f"#{tag}".lower() not in (t.lower() for t in out):
            out.append(f"#{tag}")
    if "#shorts" not in (t.lower() for t in out):
        out.insert(0, "#shorts")
    return out[:MAX_HASHTAGS]


def meta_from_dict(raw: dict[str, Any]) -> ShortsMeta:
    """Clean up whatever the model returned so it's safe to paste into YouTube."""
    title = _clip_text(raw.get("shorts_title", ""), TITLE_LIMIT)
    title = re.sub(r"\s*#\w+", "", title).strip()
    alts = raw.get("alt_titles") or []
    alt_titles = [_clip_text(t, TITLE_LIMIT) for t in alts if isinstance(t, str) and t.strip()]
    description = str(raw.get("description", "")).strip()
    # Hashtags go in their own line; drop any the model left inside the text.
    description = re.sub(r"(?:^|\s)#\w+", "", description).strip()
    return ShortsMeta(
        title=title,
        alt_titles=[t for t in alt_titles if t != title][:2],
        description=description[: DESCRIPTION_LIMIT - 200],
        hashtags=normalize_hashtags(raw.get("hashtags")) if title else [],
    )


def write_meta(meta: ShortsMeta, clip_path: Path) -> Path:
    path = clip_path.with_suffix(".txt")
    path.write_text(meta.to_text(), encoding="utf-8")
    return path


def _relative_transcript(cues: list[Cue], origin: float) -> str:
    return "\n".join(
        f"{format_timestamp(max(0.0, c.start - origin), '.')[3:-4]} {c.text.replace(chr(10), ' / ')}"
        for c in cues
    )


def build_metadata_request(
    clip_cues: list[Cue],
    *,
    context_cues: list[Cue] | None = None,
    movie: str | None = None,
    language: str | None = None,
    hint: str | None = None,
    model: str,
) -> dict[str, Any]:
    origin = clip_cues[0].start
    parts = []
    if hint:
        parts.append(
            "<on_screen>\nThe transcript has no visuals. The editor describes what happens on "
            f"screen - treat it as the main subject of the clip:\n{hint}\n</on_screen>"
        )
    if context_cues:
        parts.append(
            "<context>\nWhat happens right before the clip (for understanding only, "
            "it is NOT in the clip):\n"
            f"{_relative_transcript(context_cues, context_cues[0].start)}\n</context>"
        )
    parts.append(f"<clip_transcript>\n{_relative_transcript(clip_cues, origin)}\n</clip_transcript>")
    task = [
        "Write YouTube Shorts metadata for this clip. The transcript comes from automatic "
        "speech recognition and may contain misheard words - don't build the title on a "
        "line that looks garbled."
    ]
    task.append(f"Movie / series: {movie}." if movie else "The movie name is unknown - do not guess it.")
    task.append(f"Write in {language}." if language else "Write in the language of the transcript.")
    parts.append("\n".join(task))

    return {
        "model": model,
        "max_tokens": 1500,
        "system": (
            "You are a YouTube Shorts growth editor for a channel that posts movie clips.\n"
            + SHORTS_GUIDE
            + "\nAlways answer by calling the `submit_metadata` tool."
        ),
        "tools": [METADATA_TOOL],
        "messages": [{"role": "user", "content": "\n\n".join(parts)}],
    }


def describe_clip(
    clip_cues: list[Cue],
    *,
    context_cues: list[Cue] | None = None,
    movie: str | None = None,
    language: str | None = None,
    hint: str | None = None,
    model: str,
    client: Any = None,
) -> ShortsMeta:
    if not clip_cues:
        raise ValueError("The clip has no subtitles — nothing to write a title from.")
    if client is None:
        client = make_client()
    request = build_metadata_request(
        clip_cues, context_cues=context_cues, movie=movie, language=language, hint=hint,
        model=model,
    )
    meta = meta_from_dict(call_tool(client, request, "submit_metadata"))
    if not meta:
        raise RuntimeError("Claude did not return a title and description.")
    return meta
