from clipsmith import subtitles
from clipsmith.subtitles import format_timestamp, parse, slice_cues, to_srt

SRT = """\
1
00:00:01,000 --> 00:00:03,500
<i>Hello</i> there.

2
00:00:04,000 --> 00:00:06,000
General Kenobi!
You are a bold one.

3
00:01:00,250 --> 00:01:02,000
{\\an8}Kill him.
"""

VTT = """\
WEBVTT

00:01.000 --> 00:03.500 align:start
Hello there.

intro
01:00:04.000 --> 01:00:06.000
Second cue
"""


def test_parse_srt():
    cues = parse(SRT)
    assert [c.index for c in cues] == [1, 2, 3]
    assert cues[0].start == 1.0 and cues[0].end == 3.5
    assert cues[0].text == "Hello there."
    assert cues[1].text == "General Kenobi!\nYou are a bold one."
    assert cues[2].text == "Kill him."
    assert cues[2].start == 60.25


def test_parse_vtt():
    cues = parse(VTT)
    assert len(cues) == 2
    assert cues[0].start == 1.0
    assert cues[1].start == 3604.0
    assert cues[1].text == "Second cue"


def test_parse_crlf_and_sorting():
    text = SRT.replace("\n", "\r\n")
    reordered = "\r\n\r\n".join(reversed(text.strip().split("\r\n\r\n")))
    cues = parse(reordered)
    assert [c.text for c in cues][0] == "Hello there."


def test_load_cp1251(tmp_path):
    path = tmp_path / "ru.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nПривет, мир\n".encode("cp1251"))
    assert subtitles.load(path)[0].text == "Привет, мир"


def test_load_utf16_with_bom(tmp_path):
    path = tmp_path / "u16.srt"
    path.write_bytes("1\n00:00:01,000 --> 00:00:02,000\nПривет\n".encode("utf-16"))
    assert subtitles.load(path)[0].text == "Привет"


def test_format_timestamp():
    assert format_timestamp(3723.456) == "01:02:03,456"
    assert format_timestamp(0.9999) == "00:00:01,000"
    assert format_timestamp(-1) == "00:00:00,000"


def test_slice_and_roundtrip():
    cues = parse(SRT)
    sliced = slice_cues(cues, 2.0, 5.0)
    assert [(c.start, c.end) for c in sliced] == [(0.0, 1.5), (2.0, 3.0)]
    again = parse(to_srt(sliced))
    assert [c.text for c in again] == [c.text for c in sliced]
