import os
from types import SimpleNamespace

from clipsmith import cli
from clipsmith.metadata import (
    TITLE_LIMIT,
    ShortsMeta,
    build_metadata_request,
    describe_clip,
    meta_from_dict,
    normalize_hashtags,
    write_meta,
)
from clipsmith.subtitles import Cue

CUES = [Cue(1, 100.0, 102.0, "Кто ты?"), Cue(2, 103.0, 105.0, "Бомж.\nНикто.")]


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.request = None
        self.messages = self

    def create(self, **request):
        self.request = request
        block = SimpleNamespace(type="tool_use", name="submit_metadata", input=self.payload)
        return SimpleNamespace(content=[block])


def test_hashtags_are_cleaned_deduplicated_and_start_with_shorts():
    tags = normalize_hashtags(["кино", "#Кино", "#драма!", "русские сериалы", "a", "b", "c"])
    assert tags[0] == "#shorts"
    assert tags[1:3] == ["#кино", "#драма"]
    assert "#русскиесериалы" in tags
    assert len(tags) == 5


def test_meta_from_dict_enforces_youtube_rules():
    meta = meta_from_dict({
        "shorts_title": "Он пришёл не к той двери #shorts " + "очень " * 30,
        "alt_titles": ["Кто ты такой?", "", "Он пришёл не к той двери"],
        "description": "Кто прав? #кино\nПиши в комментариях.",
        "hashtags": ["#shorts", "#сериал"],
    })
    assert len(meta.title) <= TITLE_LIMIT and meta.title.endswith("…")
    assert "#" not in meta.title
    assert "#" not in meta.description and "Пиши в комментариях." in meta.description
    assert meta.alt_titles == ["Кто ты такой?", "Он пришёл не к той двери"]
    assert meta.full_description().endswith("#shorts #сериал")


def test_alt_title_equal_to_title_is_dropped():
    meta = meta_from_dict({"shorts_title": "Кто ты?", "alt_titles": ["Кто ты?", "Никто"]})
    assert meta.alt_titles == ["Никто"]


def test_write_meta_creates_txt_next_to_clip(tmp_path):
    meta = ShortsMeta("Заголовок", ["Альтернатива"], "Описание", ["#shorts"])
    path = write_meta(meta, tmp_path / "01-clip.mp4")
    assert path.name == "01-clip.txt"
    text = path.read_text(encoding="utf-8")
    assert "Заголовок" in text and "Альтернатива" in text and "Описание\n\n#shorts" in text


def test_request_uses_clip_relative_times_and_context():
    req = build_metadata_request(
        CUES, context_cues=[Cue(1, 40.0, 41.0, "Раньше")], movie="Сериал", model="m"
    )
    text = req["messages"][0]["content"]
    assert "00:00 Кто ты?" in text and "00:03 Бомж. / Никто." in text
    assert "<context>" in text and "Раньше" in text
    assert "Movie / series: Сериал." in text
    assert req["tool_choice"]["name"] == "submit_metadata"


def test_request_includes_on_screen_hint():
    text = build_metadata_request(CUES, hint="драка в подъезде", model="m")["messages"][0]["content"]
    assert "<on_screen>" in text and "драка в подъезде" in text


def test_request_tells_model_not_to_guess_movie():
    text = build_metadata_request(CUES, model="m")["messages"][0]["content"]
    assert "do not guess" in text


def test_describe_clip_with_fake_client():
    client = FakeClient({
        "shorts_title": "Бомж против бандита",
        "alt_titles": ["Кто ты?", "Он никто. Или нет?"],
        "description": "Чем закончится?",
        "hashtags": ["#сериал"],
    })
    meta = describe_clip(CUES, model="test-model", client=client)
    assert client.request["model"] == "test-model"
    assert meta.title == "Бомж против бандита"
    assert meta.hashtags == ["#shorts", "#сериал"]


def test_load_dotenv_does_not_override_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport CLIPSMITH_T1='from-file'\nCLIPSMITH_T2=file\n")
    monkeypatch.delenv("CLIPSMITH_T1", raising=False)
    monkeypatch.setenv("CLIPSMITH_T2", "real")
    cli.load_dotenv(env)
    assert os.environ["CLIPSMITH_T1"] == "from-file"
    assert os.environ["CLIPSMITH_T2"] == "real"
    monkeypatch.delenv("CLIPSMITH_T1")
