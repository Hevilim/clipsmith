from types import SimpleNamespace

from clipsmith.analyze import build_request, clips_from_response, suggest_clips
from clipsmith.subtitles import Cue

CUES = [Cue(i, i * 10.0, i * 10.0 + 5, f"line {i}") for i in range(1, 21)]


class FakeClient:
    def __init__(self, clips):
        self.clips = clips
        self.request = None
        self.messages = self

    def create(self, **request):
        self.request = request
        block = SimpleNamespace(type="tool_use", name="submit_clips", input={"clips": self.clips})
        return SimpleNamespace(content=[block])


def test_build_request_includes_constraints():
    req = build_request(
        CUES, count=3, min_duration=15, max_duration=60,
        theme="funny", language="Russian", model="m",
    )
    assert req["tool_choice"] == {"type": "tool", "name": "submit_clips"}
    transcript, task = req["messages"][0]["content"]
    assert "[1] 00:00:10 line 1" in transcript["text"]
    assert "up to 3 clips" in task["text"]
    assert "funny" in task["text"] and "Russian" in task["text"]


def test_clips_from_response_validates_and_sorts():
    raw = [
        {"title": "A", "start_cue": 2, "end_cue": 4, "reason": "r", "score": 6},
        {"title": "B", "start_cue": 8, "end_cue": 6, "reason": "swapped", "score": 9},
        {"title": "bad index", "start_cue": 99, "end_cue": 100, "score": 10},
        {"title": "too long", "start_cue": 1, "end_cue": 20, "score": 10},
        {"title": "garbage", "start_cue": "x"},
    ]
    clips = clips_from_response(raw, CUES, max_duration=60)
    assert [c.title for c in clips] == ["B", "A"]
    assert (clips[0].start, clips[0].end) == (60.0, 85.0)
    assert (clips[1].start_cue, clips[1].end_cue) == (2, 4)


def test_suggest_clips_with_fake_client():
    client = FakeClient([{"title": "T", "start_cue": 1, "end_cue": 3, "reason": "r", "score": 7}])
    clips = suggest_clips(CUES, client=client, model="test-model")
    assert client.request["model"] == "test-model"
    assert clips[0].title == "T" and clips[0].end == 35.0
