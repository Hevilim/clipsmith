import pytest

from clipsmith.transcribe import Word, words_to_cues


def w(start, end, text):
    return Word(start, end, text)


def test_splits_on_sentence_end():
    cues = words_to_cues([
        w(0.0, 0.3, " Стой."), w(0.4, 0.7, " Ты"), w(0.7, 1.0, " куда?"), w(1.1, 1.4, " Домой."),
    ])
    assert [c.text for c in cues] == ["Стой.", "Ты куда?", "Домой."]
    assert [c.index for c in cues] == [1, 2, 3]


def test_splits_on_pause_and_length():
    words = [w(i * 0.3, i * 0.3 + 0.25, "слово") for i in range(6)]
    words.append(w(5.0, 5.3, "после"))
    cues = words_to_cues(words, max_chars=17)
    assert [c.text for c in cues] == ["слово слово слово", "слово слово слово", "после"]


def test_splits_on_max_duration():
    words = [w(i * 1.0, i * 1.0 + 0.9, "а") for i in range(6)]
    cues = words_to_cues(words, max_duration=2.5, max_gap=5)
    assert all(c.duration <= 2.5 for c in cues)


def test_short_cue_is_extended_but_not_into_next():
    cues = words_to_cues([w(0.0, 0.2, "Да."), w(0.5, 0.8, "Нет.")], min_duration=0.7)
    assert cues[0].end == pytest.approx(0.5)
    assert cues[1].end == pytest.approx(1.2)


def test_merges_tiny_tail_into_previous_cue():
    cues = words_to_cues([
        w(0.0, 0.8, "Любишь"), w(0.8, 1.6, "женщин"), w(1.6, 2.4, "отбивать,"),
        w(2.5, 2.6, "да?"), w(2.6, 3.5, "Это"), w(3.5, 4.5, "моя."),
    ], max_chars=30)
    assert [c.text for c in cues] == ["Любишь женщин отбивать, да?", "Это моя."]


def test_does_not_merge_across_sentences():
    cues = words_to_cues([w(0.0, 1.0, "Стой."), w(1.1, 1.2, "Нет.")])
    assert [c.text for c in cues] == ["Стой.", "Нет."]


def test_skips_empty_words():
    assert words_to_cues([w(0, 1, "  ")]) == []
