"""不调用任何 TTS / Whisper 的测试：合成用假的 speak，对齐用假的 transcriber。"""

import json
import math
import struct
import wave
from pathlib import Path

from voice import align, script, synth, textalign


def tone(seconds: float, rate: int = 24000) -> bytes:
    """一段正弦波，当作 24k s16le PCM 返回（和 OpenRouter 的 pcm 格式一样）。"""
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * k / rate))) for k in range(n))


def test_split_marks_and_keys():
    spoken, marks = script.split_marks("先{@a}五减二，{@b}得三。")
    assert spoken == "先五减二，得三。" and marks == {"a": 1, "b": 5}
    doc = {"beats": [{"id": "B", "say": [{"pace": "normal", "text": "x{@m}y"}, {"id": "s2", "pace": "slow", "text": "z"}]}]}
    beats = script.beats(doc)
    assert [s.key for s in beats[0].sentences] == ["0", "s2"]


def test_textalign_matches_and_interpolates():
    heard = [textalign.HeardChar(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate("先五减2得三")]
    al = textalign.align("先五减二，得三。", heard)
    assert al.ratio == 1.0  # 阿拉伯数字按汉字比较
    assert al.time_at(textalign.content_index("先五减二，得三。", 5), 2.0) == (0.8, "matched")
    gap = [textalign.HeardChar(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate("先五得三")]
    al2 = textalign.align("先五减二得三", gap)
    t, how = al2.time_at(2, 1.0)
    assert how == "interpolated" and 0.2 < t < 0.6


def test_diff_finds_dropped_and_invented_words():
    d = textalign.diff("三十六，后面省略", "三十六，省略")
    assert d.missing == ["后面"] and d.extra == []
    d2 = textalign.diff("加三得六", "加三得六，对吧")
    assert d2.extra == ["对吧"]


def test_chars_from_words_spreads_time():
    chars = textalign.chars_from_words([{"word": "下面", "start": 1.0, "end": 1.4}, {"word": "，", "start": 1.4, "end": 1.5}])
    assert [(c.ch, c.start) for c in chars] == [("下", 1.0), ("面", 1.2)]


def test_synth_beat_with_fake_speaker(tmp_path):
    calls = []

    def speak(instructions, text, model):
        calls.append(text)
        return tone(0.5 + 0.05 * len(text))

    beat = script.beats({"beats": [{"id": "B", "say": [
        {"pace": "normal", "text": "第一句{@a}。"}, {"pace": "ask", "text": "第二句？"}]}]})[0]
    timing = synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=speak, log=lambda *_: None)
    assert calls == ["第一句。", "第二句？"]  # 书签不念
    assert timing["cues"][0]["start"] == synth.LEAD
    gap = timing["cues"][1]["start"] - timing["cues"][0]["end"]
    assert abs(gap - synth.GAP["normal"]) < 0.01  # 停顿由说完的那一句的语气决定
    with wave.open(str(tmp_path / "B.wav")) as w:
        assert abs(w.getnframes() / w.getframerate() - timing["total"]) < 0.05
    # 第二次走缓存，不再调用
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=speak, log=lambda *_: None)
    assert len(calls) == 2


def test_align_beat_with_fake_transcriber(tmp_path):
    beat = script.beats({"beats": [{"id": "B", "say": [{"id": "s1", "pace": "normal", "text": "一二{@a}三四"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=lambda *a: tone(1.0), log=lambda *_: None)

    def transcriber(samples, prompt, model):
        words = [{"word": ch, "start": 0.15 + i * 0.25, "end": 0.4 + i * 0.25} for i, ch in enumerate("一二三四")]
        return "一二三四", words

    results = align.align_beat(beat, tmp_path, transcriber=transcriber, log=lambda *_: None)
    assert results[0].how == {"a": "matched"} and not results[0].flagged
    data = json.loads((tmp_path / "B.timing.json").read_text())
    start = data["cues"][0]["start"]
    assert abs(data["marks"]["s1"]["a"] - (start - align.PAD + 0.65)) < 1e-6
    assert data["check"]["s1"]["ratio"] == 1.0


def test_tail_relisten_when_word_stamps_stop_early(tmp_path):
    beat = script.beats({"beats": [{"id": "B", "say": [{"id": "s1", "pace": "normal", "text": "一二三四五六{@z}七八"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=lambda *a: tone(3.0), log=lambda *_: None)
    calls = []

    def transcriber(samples, prompt, model):
        calls.append(prompt)
        if len(calls) == 1:  # 第一遍只给出前三个字的时间戳
            return "一二三四五六七八", [{"word": ch, "start": i * 0.3, "end": i * 0.3 + 0.3} for i, ch in enumerate("一二三")]
        return "", [{"word": ch, "start": 0.1 + i * 0.3, "end": 0.4 + i * 0.3} for i, ch in enumerate(prompt)]

    results = align.align_beat(beat, tmp_path, transcriber=transcriber, log=lambda *_: None)
    assert calls[1] == "四五六七八"  # 只补听没覆盖到的部分
    assert results[0].how["z"] == "matched"


def test_cli_align_requires_audio(tmp_path):
    from voice.cli import main

    spec = Path(__file__).resolve().parents[2] / "scene-spec" / "examples" / "s67-p01.scene.yaml"
    try:
        main(["align", str(spec), "--audio", str(tmp_path), "--only", "Down"])
    except FileNotFoundError as exc:
        assert "synth" in str(exc)
    else:
        raise AssertionError("缺少音频时应报错")


def test_usage_is_logged_with_cost(tmp_path):
    beat = script.beats({"beats": [{"id": "B", "say": [{"pace": "normal", "text": "一句话"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=lambda *a: tone(2.0), log=lambda *_: None)
    calls, usd = synth.total_cost(tmp_path / "tts_usage.jsonl")
    assert calls == 1 and 0.0003 < usd < 0.002  # 2 秒音频约 $0.00045，加上输入 token


def test_confirm_drops_whisper_only_misses(tmp_path):
    beat = script.beats({"beats": [{"id": "B", "say": [{"pace": "normal", "text": "后面省略而且"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=lambda *a: tone(1.5), log=lambda *_: None)
    raw = tmp_path / "lines" / "B-0.wav"
    # 不带提示词时漏了"后面"，带提示词时听全了 → 不算确认缺失
    answers = iter(["省略而且", "后面省略而且"])
    assert align.confirm(beat.sentences[0], raw, "m", lambda s, p, m: (next(answers), [])) == []
    # 两次都没听到"动作" → 确认缺失
    beat2 = script.beats({"beats": [{"id": "B", "say": [{"pace": "normal", "text": "用这个动作去证明"}]}]})[0]
    answers = iter(["用这个递回去证明", "用这个递回去证明"])
    assert align.confirm(beat2.sentences[0], raw, "m", lambda s, p, m: (next(answers), [])) == ["动作"]
