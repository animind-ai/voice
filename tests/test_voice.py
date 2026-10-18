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

    def transcriber(samples, prompt, model, language="zh"):
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

    def transcriber(samples, prompt, model, language="zh"):
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
    assert align.confirm(beat.sentences[0], raw, "m", lambda s, p, m, lang="zh": (next(answers), [])) == []
    # 两次都没听到"动作" → 确认缺失
    beat2 = script.beats({"beats": [{"id": "B", "say": [{"pace": "normal", "text": "用这个动作去证明"}]}]})[0]
    answers = iter(["用这个递回去证明", "用这个递回去证明"])
    assert align.confirm(beat2.sentences[0], raw, "m", lambda s, p, m, lang="zh": (next(answers), [])) == ["动作"]


def test_backend_choice(monkeypatch):
    monkeypatch.delenv("ANIMIND_ALIGN_BACKEND", raising=False)
    monkeypatch.setattr(align.sys, "platform", "linux")
    assert align.default_backend() == "faster"
    monkeypatch.setattr(align.sys, "platform", "darwin")
    assert align.default_backend() == "mlx"
    monkeypatch.setenv("ANIMIND_ALIGN_BACKEND", "faster")
    assert align.default_backend() == "faster"
    monkeypatch.setenv("ANIMIND_ALIGN_BACKEND", "cuda")
    try:
        align.default_backend()
    except ValueError:
        pass
    else:
        raise AssertionError("未知后端应报错")


def test_faster_output_matches_mlx_shape(monkeypatch):
    """faster-whisper 的 Segment / Word 对象转成和 mlx 一样的 dict，textalign 能直接用。"""
    from types import SimpleNamespace as NS

    seen = {}

    class FakeModel:
        def transcribe(self, samples, **kw):
            seen.update(kw)
            words = [NS(word=ch, start=i * 0.2, end=i * 0.2 + 0.2, probability=0.9) for i, ch in enumerate("一二三")]
            return iter([NS(text="一二", words=words[:2]), NS(text="三", words=words[2:])]), None

    monkeypatch.setitem(align._faster_models, "m", FakeModel())
    text, words = align.transcribe_faster(None, "", "m")
    assert text == "一二三" and seen["initial_prompt"] is None and seen["word_timestamps"]
    assert [w["word"] for w in words] == ["一", "二", "三"] and words[2]["start"] == 0.4
    assert textalign.align("一二三", textalign.chars_from_words(words)).ratio == 1.0


def hz_tone(freq: float, seconds: float = 0.6, rate: int = 24000) -> bytes:
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * k / rate))) for k in range(n))


def test_pitch_tells_low_voice_from_high(tmp_path):
    from voice import pitch

    for freq in (120, 210):
        synth._to_wav(hz_tone(freq), tmp_path / f"{freq}.wav")
    low, high = pitch.median_f0(tmp_path / "120.wav"), pitch.median_f0(tmp_path / "210.wav")
    assert abs(low - 120) < 6 and abs(high - 210) < 10
    assert pitch.off_voice(low, "Kore") and not pitch.off_voice(high, "Kore")
    assert pitch.off_voice(high, "Puck") and not pitch.off_voice(low, "Puck")
    assert not pitch.off_voice(low, "unknown-voice")  # 不认识的声音不做判断


def test_synth_retries_when_voice_changes(tmp_path):
    takes = iter([hz_tone(120), hz_tone(210)])
    calls = []

    def speak(instructions, text, model):
        calls.append(instructions)
        return next(takes)

    wav = synth.synth_line(tmp_path / "lines", "B-0", "normal", "一句话。", "teacher-hs", "fake/model", "Kore",
                           speak=speak, log=lambda *_: None)
    from voice import pitch

    assert len(calls) == 2  # 第一次像男声，重合成一次
    assert pitch.median_f0(wav) > pitch.FEMALE_FLOOR
    assert synth.STEADY in calls[0]
    assert len((tmp_path / "tts_usage.jsonl").read_text().splitlines()) == 2  # 两次都记账


def test_synth_keeps_closest_take_after_retries(tmp_path):
    takes = iter([hz_tone(110), hz_tone(140), hz_tone(120)])
    synth.synth_line(tmp_path / "lines", "B-0", "normal", "一句话。", "teacher-hs", "fake/model", "Kore",
                     speak=lambda *a: next(takes), log=lambda *_: None)
    from voice import pitch

    assert abs(pitch.median_f0(tmp_path / "lines" / "B-0.wav") - 140) < 8


def test_loose_cache_ignores_changed_instructions(tmp_path):
    calls = []

    def speak(instructions, text, model):
        calls.append(text)
        return tone(0.5)

    beat = script.beats({"beats": [{"id": "B", "say": [
        {"pace": "normal", "text": "第一句。"}, {"pace": "normal", "text": "第二句。"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=speak, log=lambda *_: None)
    stamp = tmp_path / "lines" / "B-0.json"
    old = json.loads(stamp.read_text())
    stamp.write_text(json.dumps(dict(old, instructions="旧的人设"), ensure_ascii=False))
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=speak, log=lambda *_: None,
                     force={"B-1"}, loose=True)
    assert calls == ["第一句。", "第二句。", "第二句。"]  # 只重念强制的那句，旧人设的第一句沿用缓存


# ---------- 英文版（style.lang: en） ----------

def test_english_sentences_skip_zh_normalize_and_keep_marks():
    doc = {"style": {"lang": "en"}, "beats": [{"id": "B", "say": [
        {"pace": "normal", "text": "{@a} side a is two and {@b}side b is three."}]}]}
    s = script.beats(doc)[0].sentences[0]
    assert s.spoken == "side a is two and side b is three." and s.lang == "en"
    assert s.spoken[s.marks["a"]:].startswith("side a") and s.spoken[s.marks["b"]:].startswith("side b")
    assert s.warnings() == []


def test_english_instructions():
    text = synth.instructions_for("teacher-hs", "ask", "en")
    assert text.isascii() and "exactly as written" in text and "questioning" in text
    assert synth.instructions_for("teacher-hs", "ask") != text


def test_spell_numbers_for_english_alignment():
    assert textalign.spell_numbers("side a is 2, 3.5 and 1,024") == \
        "side a is two, three point five and one thousand twenty four"
    heard = textalign.chars_from_words([{"word": " two", "start": 1.0, "end": 1.3}])
    al = textalign.align("is two", heard)
    assert al.time_at(textalign.content_index("is two", 3), 2.0) == (1.0, "matched")


def test_align_records_word_start_times(monkeypatch):
    monkeypatch.setattr(align, "slice_16k", lambda *a: None)
    doc = {"style": {"lang": "en"}, "beats": [{"id": "B", "say": [{"pace": "normal", "text": "one two"}]}]}
    sentence = script.beats(doc)[0].sentences[0]
    words = [{"word": " one", "start": 0.1, "end": 0.4}, {"word": " two", "start": 0.9, "end": 1.2}]
    res = align.align_sentence(sentence, Path("unused.wav"), 0.5, 2.0, "m",
                               transcriber=lambda *a, **k: ("one two", words))
    assert res.content == 6 and res.words == [[0, 0.45], [3, 1.25]]
