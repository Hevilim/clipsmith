import numpy as np
import pytest

from clipsmith import music
from clipsmith.analyze import Clip, build_request, drop_music_clips, format_transcript
from clipsmith.subtitles import Cue

# hop 5 s: music in 10-20 s and 40-45 s
MAP = music.MusicMap(hop=5.0, scores=[0.1, 0.1, 0.9, 0.8, 0.0, 0.2, 0.0, 0.0, 0.7, 0.1])


def test_fraction():
    assert MAP.fraction(0, 10) == 0
    assert MAP.fraction(10, 20) == 1
    assert MAP.fraction(5, 25) == pytest.approx(0.5)
    assert MAP.fraction(12.5, 17.5) == 1
    assert MAP.fraction(0, 10, threshold=0.05) == 1


def test_segments_merge_close_gaps():
    assert MAP.segments() == [(10.0, 20.0), (40.0, 45.0)]
    assert MAP.segments(min_gap=20) == [(10.0, 45.0)]


def test_save_and_load(tmp_path):
    path = tmp_path / "film.music.json"
    MAP.save(path)
    assert music.MusicMap.load(path).scores == MAP.scores
    assert music.map_path(tmp_path / "film.mp4").name == "film.music.json"


def test_model_input_shape_and_normalisation():
    t = np.arange(int(3 * music.SAMPLE_RATE)) / music.SAMPLE_RATE
    x = music.model_input((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert x.shape == (1024, 128) and x.dtype == np.float32
    # Like ASTFeatureExtractor: zero padding after 3 s of audio, applied before normalising.
    assert np.allclose(x[400:], 4.2677393 / (2 * 4.5689974))
    # A 440 Hz tone should light up a low mel band, not the top of the range.
    assert x[50].argmax() < 40


def test_music_probability_takes_max_over_music_classes():
    logits = np.full((1, 527), -10.0)
    logits[0, 0] = 5.0  # Speech
    logits[0, 270] = 0.0  # Soundtrack music -> sigmoid 0.5
    assert music.music_probability(logits) == [pytest.approx(0.5)]


def test_list_tracks_and_credit(tmp_path):
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "a.ogg").write_bytes(b"")
    (tmp_path / "a.txt").write_text("Music: Someone (CC BY 4.0)\n")
    tracks = music.list_tracks(tmp_path)
    assert [t.name for t in tracks] == ["a.ogg", "b.mp3"]
    assert music.track_credit(tracks[0]) == "Music: Someone (CC BY 4.0)"
    assert music.track_credit(tracks[1]) == ""
    with pytest.raises(FileNotFoundError):
        music.list_tracks(tmp_path / "missing.mp3")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        music.list_tracks(empty)


def test_transcript_marks_cues_over_music():
    cues = [Cue(1, 2.0, 4.0, "тихо"), Cue(2, 11.0, 14.0, "под музыку")]
    text = format_transcript(cues, MAP)
    assert "[1] 00:00:02 тихо" in text
    assert "[2] 00:00:11 ♪ под музыку" in text
    req = build_request(cues, count=1, min_duration=1, max_duration=60, theme=None,
                        language=None, model="m", music_map=MAP)
    task = req["messages"][0]["content"][1]["text"]
    assert "WITHOUT ♪" in task
    assert "music-free windows: 00:00:00-00:00:10" in task


def test_music_free_windows():
    from clipsmith.analyze import music_free_windows

    assert music_free_windows(MAP, 5, 60) == [(0.0, 10.0), (20.0, 40.0), (45.0, 60.0)]
    assert music_free_windows(MAP, 16, 60) == [(20.0, 40.0)]


def test_drop_music_clips():
    quiet = Clip("quiet", 0, 10, 1, 2)
    loud = Clip("loud", 8, 22, 3, 4)
    kept, rejected = drop_music_clips([quiet, loud], MAP, max_music=0.15)
    assert kept == [quiet] and rejected == [loud]
    assert loud.music == pytest.approx(0.71, abs=0.01)
