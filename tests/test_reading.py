"""旁白读法（zh-math-reading）接入后的测试：公式展开、书签位置、已有缓存不失效、synth 前打印提示。不调用任何 TTS。"""

from pathlib import Path

import pytest
from scene_spec.loader import load_file

from voice import script, synth

EXAMPLES = Path(__file__).resolve().parents[2] / "scene-spec" / "examples"


@pytest.mark.parametrize("name", ["s67-p01", "s67-p02"])
def test_existing_examples_keep_byte_identical_spoken(name):
    """不含公式的旁白读法逐字节不变：TTS 缓存按读法文本做键，已合成的音频继续有效。"""
    doc = load_file(EXAMPLES / f"{name}.scene.yaml")
    sentences = [s for b in script.beats(doc) for s in b.sentences]
    assert len(sentences) > 30
    for s in sentences:
        spoken, marks = script.split_marks(s.text)  # 接入 zh-math-reading 之前的算法
        assert s.spoken == spoken
        assert s.marks == marks


def test_formula_is_expanded_and_bookmarks_follow():
    doc = {"beats": [{"id": "B", "say": [
        {"id": "s1", "pace": "normal", "text": "第二问，{@p2}数列 $b_n$ 从第一项起，而且 {@d}$d=1$；{@b}$b_{n+1}$ 等于二。"},
    ]}]}
    s = script.beats(doc)[0].sentences[0]
    assert s.spoken == "第二问，数列 b n 从第一项起，而且 d 等于一；b n 加一等于二。"
    assert s.spoken[s.marks["p2"]:].startswith("数列 b n")
    assert s.spoken[s.marks["d"]:].startswith("d 等于一")
    assert s.spoken[s.marks["b"]:].startswith("b n 加一")


def test_bookmark_after_formula_points_at_next_word():
    doc = {"beats": [{"id": "B", "say": [{"pace": "normal", "text": "叫{@x}$d${@y}等差比数列"}]}]}
    s = script.beats(doc)[0].sentences[0]
    assert s.spoken == "叫 d 等差比数列"
    assert s.spoken[s.marks["x"]] == "d" and s.spoken[s.marks["y"]] == "等"


def test_naming_colon_is_rewritten():
    doc = {"beats": [{"id": "B", "say": [{"pace": "normal", "text": "又造了一个新名词：{@n}d 等差比数列。"}]}]}
    s = script.beats(doc)[0].sentences[0]
    assert s.spoken == "又造了一个新名词，叫 d 等差比数列。"
    assert s.spoken[s.marks["n"]] == "d"


def test_synth_sends_expanded_text(tmp_path):
    calls = []

    def speak(instructions, text, model):
        calls.append(text)
        return b"\x00\x00" * 2400

    beat = script.beats({"beats": [{"id": "B", "say": [{"pace": "normal", "text": "所以 {@a}$A_8=121$。"}]}]})[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "fake/model", "v", speak=speak, log=lambda *_: None)
    assert calls == ["所以 A 八等于一百二十一。"]


def test_sentence_warnings():
    doc = {"beats": [{"id": "B", "say": [
        {"pace": "normal", "text": "得到：{@r}b n 加 b n 加二"},
        {"pace": "normal", "text": "公差为一的等差数列。"},
    ]}]}
    a, b = script.beats(doc)[0].sentences
    assert any("后紧跟字母「b」" in w for w in a.warnings())
    assert b.warnings() == []


def test_cli_synth_prints_lint_before_tts(tmp_path, monkeypatch, capsys):
    from voice.cli import main

    order = []

    def fake_synth_beat(beat, *args, **kwargs):  # 不调用真实 TTS
        order.append(("synth", beat.id))
        return {}

    monkeypatch.setattr(synth, "synth_beat", fake_synth_beat)
    real_print = print

    def tracking_print(*args, **kwargs):
        if args and "⚠" in str(args[0]):
            order.append(("warn", str(args[0])))
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", tracking_print)
    code = main(["synth", str(EXAMPLES / "s67-p02.scene.yaml"), "--out", str(tmp_path), "--model", "fake/m", "--voice", "v",
                 "--only", "Prove"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Prove[s3]" in out and "读法检查" in out
    assert order[0][0] == "warn" and order[-1] == ("synth", "Prove")
