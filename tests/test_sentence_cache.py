"""句级 TTS 缓存（.cache/tts/）：只改一句只重念一句，没改的句子音频逐字节不变。全部用假的 speak，不联网。"""

import json
import math
import os
import struct
import threading
import time
import wave
from pathlib import Path

import pytest

from voice import script, synth, ttscache
from voice.cli import main

SAY = [
    {"id": "s1", "pace": "normal", "text": "第一句。"},
    {"id": "s2", "pace": "normal", "text": "第二句，", "hold": 0.3},
    {"id": "s3", "pace": "ask", "text": "第三句？"},
]


def voice_like(text: str, rate: int = 24000) -> bytes:
    """每段文字一个固定的、Kore 音区里的音高和时长：同一句念两遍得到一样的字节，不同的句子不一样。"""
    freq = 200 + sum(map(ord, text)) % 30
    n = int((0.4 + 0.04 * len(text)) * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * k / rate))) for k in range(n))


class Speaker:
    def __init__(self):
        self.calls = []

    def __call__(self, instructions, text, model):
        self.calls.append(text)
        return voice_like(text)


def run(beat, out, speak, logs=None, model="google/gemini-3.8-flash-tts", voice="Kore", speed=1.0, **kw):
    return synth.synth_beat(beat, out, "teacher-hs", model, voice, speed, speak=speak,
                            log=(logs.append if logs is not None else lambda *_: None), **kw)


def beat_of(say):
    return script.beats({"beats": [{"id": "Hook", "say": say}]})[0]


def frames(path):
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def sentence_at(beat_pcm: bytes, start: float, raw_pcm: bytes) -> bytes:
    """beat 音频里从 cue 起点开始、和原声一样长的那一段（按 cue 换算采样点，容许 ±2 个采样的毫秒圆整）。"""
    centre = round(start * synth.RATE)
    for shift in (0, -1, 1, -2, 2):
        i = 2 * (centre + shift)
        if beat_pcm[i:i + len(raw_pcm)] == raw_pcm:
            return raw_pcm
    i = 2 * centre
    return beat_pcm[i:i + len(raw_pcm)]


@pytest.fixture
def no_sleep(monkeypatch):
    """缓存命中不应该等待任何限速；有人 sleep 就失败。"""
    def forbidden(seconds):
        raise AssertionError(f"不应 sleep({seconds})")

    monkeypatch.setattr(time, "sleep", forbidden)


def test_edit_one_sentence_only_that_sentence_is_synthesized(tmp_path, no_sleep):
    speak = Speaker()
    first = run(beat_of(SAY), tmp_path, speak)
    assert len(speak.calls) == 3
    old_pcm = frames(tmp_path / "Hook.wav")
    raws = {s["id"]: frames(tmp_path / "lines" / f"Hook-{i}.wav") for i, s in enumerate(SAY)}

    edited = [dict(s) for s in SAY]
    edited[1]["text"] = "第二句改得长了一点，"  # 改中间那句：第三句在 beat 里的位置会后移
    speak.calls.clear()
    logs = []
    second = run(beat_of(edited), tmp_path, speak, logs)

    assert speak.calls == ["第二句改得长了一点，"]  # 只调用一次 TTS
    assert any(line.startswith("Hook: 2 cached, 1 synthesized") for line in logs), logs
    new_pcm = frames(tmp_path / "Hook.wav")
    for k in (0, 2):  # 没改的两句：按 cue 位置取出的采样和改之前完全一样，也和缓存里的原声一样
        raw = raws[SAY[k]["id"]]
        assert sentence_at(new_pcm, second["cues"][k]["start"], raw) == raw
        assert sentence_at(old_pcm, first["cues"][k]["start"], raw) == raw
    assert second["cues"][2]["start"] > first["cues"][2]["start"]
    gap = second["cues"][2]["start"] - second["cues"][1]["end"]
    assert abs(gap - 0.3) < 0.002  # 拼接用的是改后的那句的 hold


def test_unchanged_beat_is_not_reassembled_unless_forced(tmp_path, no_sleep):
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    timing_path = tmp_path / "Hook.timing.json"
    aligned = json.loads(timing_path.read_text())
    aligned["marks"] = {"s1": {"a": 0.5}}  # align 写回的书签：没变的 beat 不能被冲掉
    timing_path.write_text(json.dumps(aligned))

    logs = []
    run(beat_of(SAY), tmp_path, speak, logs)
    assert len(speak.calls) == 3 and "沿用" in logs[-1] and logs[-1].startswith("Hook: 3 cached, 0 synthesized")
    assert json.loads(timing_path.read_text())["marks"] == {"s1": {"a": 0.5}}

    logs.clear()
    run(beat_of(SAY), tmp_path, speak, logs, reassemble=True)  # --force：重拼，句子仍读缓存
    assert len(speak.calls) == 3 and logs[-1].startswith("Hook: 3 cached, 0 synthesized") and "沿用" not in logs[-1]
    assert "marks" not in json.loads(timing_path.read_text())


@pytest.mark.parametrize("change", [{"voice": "Aoede"}, {"model": "google/other-tts"}])
def test_voice_or_model_change_invalidates(tmp_path, change):
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    logs = []
    run(beat_of(SAY), tmp_path, speak, logs, **change)
    assert len(speak.calls) == 6 and logs[-1].startswith("Hook: 0 cached, 3 synthesized")


def test_pace_change_invalidates_only_that_sentence(tmp_path):
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    edited = [dict(s) for s in SAY]
    edited[0]["pace"] = "slow"
    logs = []
    run(beat_of(edited), tmp_path, speak, logs)
    assert speak.calls[3:] == ["第一句。"] and logs[-1].startswith("Hook: 2 cached, 1 synthesized")


def test_speed_change_reassembles_without_calling_tts(tmp_path):
    """speed 是本地 atempo：变速音频按 (原声 sha256, speed) 另存，改语速不重新花钱念。"""
    speak = Speaker()
    slow = run(beat_of(SAY), tmp_path, speak)
    logs = []
    fast = run(beat_of(SAY), tmp_path, speak, logs, speed=1.25)
    assert len(speak.calls) == 3 and logs[-1].startswith("Hook: 3 cached, 0 synthesized") and "沿用" not in logs[-1]
    assert fast["cues"][0]["end"] < slow["cues"][0]["end"]
    metas = [json.loads(p.read_text()) for p in ttscache.tts_dir(tmp_path).glob("*.json")]
    assert sorted(m["kind"] for m in metas) == ["speed"] * 3 + ["tts"] * 3
    # 回到 1.0：原声还在缓存里，也不调用
    back = run(beat_of(SAY), tmp_path, speak, speed=1.0)
    assert len(speak.calls) == 3 and back["cues"] == slow["cues"]


def test_no_cache_resynthesizes_every_sentence_and_refreshes(tmp_path):
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    logs = []
    run(beat_of(SAY), tmp_path, speak, logs, no_cache=True)
    assert len(speak.calls) == 6 and logs[-1].startswith("Hook: 0 cached, 3 synthesized")
    logs.clear()
    run(beat_of(SAY), tmp_path, speak, logs)  # 刷新后的缓存照样能命中
    assert len(speak.calls) == 6 and logs[-1].startswith("Hook: 3 cached, 0 synthesized")


def _entry(tmp_path, text):
    for meta_path in ttscache.tts_dir(tmp_path).glob("*.json"):
        meta = json.loads(meta_path.read_text())
        if meta.get("text") == text:
            return ttscache.tts_dir(tmp_path).parent / "objects" / f"{meta['sha256']}.wav", meta_path
    raise AssertionError(text)


@pytest.mark.parametrize("damage", ["truncate", "garbage_wav", "drop_meta", "bad_meta", "wrong_format"])
def test_damaged_cache_entry_is_detected_and_resynthesized(tmp_path, damage):
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    wav, meta = _entry(tmp_path, "第二句，")
    if damage == "truncate":  # 写到一半被打断：头还在，数据短了
        wav.write_bytes(wav.read_bytes()[: wav.stat().st_size // 2])
    elif damage == "garbage_wav":
        wav.write_bytes(b"RIFF....WAVEjunk")
    elif damage == "drop_meta":
        meta.unlink()
    elif damage == "bad_meta":
        meta.write_text("{not json")
    else:
        meta.write_text(json.dumps({**json.loads(meta.read_text()), "format": ttscache.FORMAT + 1}))
    logs = []
    run(beat_of(SAY), tmp_path, speak, logs)
    if damage == "drop_meta":  # 只丢了索引：lines/ 里的副本说明完全对得上，收回缓存，不重念
        assert speak.calls[3:] == [] and logs[-1].startswith("Hook: 3 cached, 0 synthesized")
    else:
        assert speak.calls[3:] == ["第二句，"]
        assert logs[-1].startswith("Hook: 2 cached, 1 synthesized")
        assert any("缓存坏了" in line for line in logs)
    hit, problem = ttscache.lookup(ttscache.tts_dir(tmp_path), json.loads(meta.read_text())["key"], synth.RATE)
    assert hit == wav and problem is None  # 重写好了


def test_pre_cache_lines_are_adopted_without_paying_again(tmp_path):
    """升级前的目录只有 lines/<beat>-<i>.wav + 说明：说明完全对得上就收进缓存，不重念。"""
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    import shutil

    shutil.rmtree(tmp_path / ".cache")
    for stamp in (tmp_path / "lines").glob("*.json"):  # 旧说明没有 key
        stamp.write_text(json.dumps({k: v for k, v in json.loads(stamp.read_text()).items() if k != "key"}))
    logs = []
    run(beat_of(SAY), tmp_path, speak, logs)
    assert len(speak.calls) == 3 and logs[-1].startswith("Hook: 3 cached, 0 synthesized")
    assert len(list(ttscache.tts_dir(tmp_path).glob("*.json"))) == 3


def test_interrupted_write_leaves_no_usable_entry(tmp_path, monkeypatch):
    calls = []

    def flaky(instructions, text, model):
        calls.append(text)
        if text == "第三句？" and len(calls) == 3:
            raise RuntimeError("网络断了")
        return voice_like(text)

    with pytest.raises(RuntimeError):
        run(beat_of(SAY), tmp_path, flaky)
    logs = []
    run(beat_of(SAY), tmp_path, flaky, logs)
    assert calls[3:] == ["第三句？"] and logs[-1].startswith("Hook: 2 cached, 1 synthesized")
    leftovers = [p.name for p in tmp_path.rglob("*") if ".tmp" in p.name]
    assert leftovers == []


SCENE = """meta:
  id: t-01
style:
  voice:
    persona: teacher-hs
    voice: Kore
    model: google/gemini-3.8-flash-tts
    speed: 1.0
beats:
  - id: Hook
    say:
      - {id: s1, pace: normal, text: "第一句。"}
      - {id: s2, pace: normal, text: "第二句，", hold: 0.3}
      - {id: s3, pace: ask, text: "第三句？"}
  - id: Next
    say:
      - {pace: normal, text: "下一段。"}
"""


def test_cli_force_and_no_cache_keep_manifest_per_beat(tmp_path, monkeypatch, capsys):
    from voice import policy

    for name in (policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")
    speak = Speaker()
    monkeypatch.setattr(synth, "default_speak", speak)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0
    assert len(speak.calls) == 4

    scene.write_text(SCENE.replace("第三句？", "第三句改了？"), encoding="utf-8")
    capsys.readouterr()
    assert main(["synth", str(scene), "--out", str(out), "--only", "Hook", "--force"]) == 0
    assert speak.calls[4:] == ["第三句改了？"]
    printed = capsys.readouterr().out
    assert "Hook: 2 cached, 1 synthesized" in printed and "调用 TTS 1 次" in printed

    assert main(["synth", str(scene), "--out", str(out), "--only", "Next", "--speed", "1.25"]) == 0
    assert len(speak.calls) == 5  # 改语速不调用 TTS
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["beats"]["Hook"] == {"model": "google/gemini-3.8-flash-tts", "voice": "Kore", "speed": 1.0, "policy_ok": True}
    assert manifest["beats"]["Next"]["speed"] == 1.25 and manifest["mixed"] is True
    assert ".cache" not in json.dumps(manifest)

    assert main(["synth", str(scene), "--out", str(out), "--no-cache"]) == 0
    assert len(speak.calls) == 9
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mixed"] is False and manifest["policy_ok"] is True and set(manifest["beats"]) == {"Hook", "Next"}


# ---------- 发布：书签移动、并发、中途打断 ----------

ALIGNED = {"marks": {"0": {"m": 0.9}}, "check": {"0": {"ratio": 1.0}}, "words": {"0": {"n": 2, "at": [[0, 0.4]]}},
           "aligner": "fake-whisper"}


def _align(tmp_path, beat_id="Hook"):
    """模拟 align：把书签时刻等写回 timing.json。"""
    path = tmp_path / f"{beat_id}.timing.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **ALIGNED}))


def _consistent(tmp_path, beat_id="Hook"):
    """发布出去的三件套互相对得上：说明里的 wav 摘要就是现在的 wav，cues 就是现在 timing.json 的 cues，时长也对。"""
    stamp = json.loads((ttscache.beats_dir(tmp_path) / f"{beat_id}.json").read_text())
    timing = json.loads((tmp_path / f"{beat_id}.timing.json").read_text())
    wav = tmp_path / f"{beat_id}.wav"
    assert stamp["wav_sha256"] == ttscache.sha256_file(wav)
    assert (stamp["cues"], stamp["total"]) == (timing["cues"], timing["total"])
    with wave.open(str(wav)) as w:
        assert abs(w.getnframes() / w.getframerate() - timing["total"]) < 0.01
    return timing


@pytest.mark.parametrize("before, after", [
    ("甲{@m}乙。", "甲乙{@m}。"),  # 只挪书签：念的字一样，原声不变
    ("甲{@m}乙。", "甲{@n}乙。"),  # 书签改名
])
def test_moving_a_bookmark_reuses_audio_but_drops_stale_alignment(tmp_path, before, after):
    speak = Speaker()
    run(beat_of([{"pace": "normal", "text": before}]), tmp_path, speak)
    _align(tmp_path)
    wav_before = ttscache.sha256_file(tmp_path / "Hook.wav")
    logs = []
    timing = run(beat_of([{"pace": "normal", "text": after}]), tmp_path, speak, logs)
    assert len(speak.calls) == 1  # 原声沿用
    assert ttscache.sha256_file(tmp_path / "Hook.wav") == wav_before
    on_disk = json.loads((tmp_path / "Hook.timing.json").read_text())
    assert set(on_disk) == {"cues", "total"} == set(timing)  # 过期的 marks / check / words / aligner 都清掉了
    assert "清掉" in logs[-1] and logs[-1].startswith("Hook: 1 cached, 0 synthesized")
    _consistent(tmp_path)
    logs.clear()
    run(beat_of([{"pace": "normal", "text": after}]), tmp_path, speak, logs)  # 之后就是普通的“没变”
    assert "沿用" in logs[-1] and "清掉" not in logs[-1]


def test_renaming_a_sentence_id_drops_stale_alignment(tmp_path):
    speak = Speaker()
    run(beat_of([{"id": "a", "pace": "normal", "text": "甲乙。"}]), tmp_path, speak)
    _align(tmp_path)
    run(beat_of([{"id": "b", "pace": "normal", "text": "甲乙。"}]), tmp_path, speak)
    assert len(speak.calls) == 1 and "marks" not in json.loads((tmp_path / "Hook.timing.json").read_text())


def test_unchanged_script_keeps_alignment(tmp_path):
    speak = Speaker()
    run(beat_of([{"pace": "normal", "text": "甲{@m}乙。"}]), tmp_path, speak)
    _align(tmp_path)
    run(beat_of([{"pace": "normal", "text": "甲{@m}乙。"}]), tmp_path, speak)
    assert json.loads((tmp_path / "Hook.timing.json").read_text())["marks"] == ALIGNED["marks"]


def _clean_env(monkeypatch):
    from voice import policy

    for name in (policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")


class ThreadSpeaker:
    """按线程给不同的音频：B 念得更长、音高不同，拼错了一眼就能从时长看出来。"""

    def __init__(self):
        self.calls = []

    def __call__(self, instructions, text, model):
        who = threading.current_thread().name
        self.calls.append(who)
        if who == "B":
            text = text * 3
        return voice_like(text)


@pytest.mark.parametrize("b_args, b_voice", [
    (["--voice", "Aoede"], "Aoede"),  # 另一个声音：manifest 必须记成最后跑完的那个
    (["--no-cache"], "Kore"),         # 同一个声音、同样的缓存键，B 刷新缓存
])
def test_concurrent_cli_synth_runs_serialize_and_end_consistent(tmp_path, monkeypatch, capsys, b_args, b_voice):
    _clean_env(monkeypatch)
    speak = ThreadSpeaker()
    monkeypatch.setattr(synth, "default_speak", speak)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    real_lock, b_waiting, results = ttscache.dir_lock, threading.Event(), {}

    def lock(audio_dir, wait=True, log=print):
        def spy(line):
            if threading.current_thread().name == "B" and line.startswith("waiting for another synth on"):
                b_waiting.set()
            log(line)
        return real_lock(audio_dir, wait, spy)

    monkeypatch.setattr(ttscache, "dir_lock", lock)
    b = threading.Thread(name="B", target=lambda: results.update(B=main(["synth", str(scene), "--out", str(out), *b_args])))
    first = speak.__call__

    def a_speak(instructions, text, model):
        if threading.current_thread().name == "A" and not b.is_alive() and "B" not in results:
            b.start()  # A 已经拿着锁在合成：这时让 B 起跑
            assert b_waiting.wait(10), "B 应该打出 waiting for another synth 并等着"
        return first(instructions, text, model)

    monkeypatch.setattr(synth, "default_speak", a_speak)
    a = threading.Thread(name="A", target=lambda: results.update(A=main(["synth", str(scene), "--out", str(out)])))
    a.start()
    a.join(60)
    b.join(60)
    assert results == {"A": 0, "B": 0}
    assert speak.calls == ["A"] * 4 + ["B"] * 4  # B 一句都没在 A 跑完之前念
    for beat_id in ("Hook", "Next"):
        timing = _consistent(out, beat_id)  # wav、timing、说明是同一套
        b_wav = timing["cues"][0]["end"] - timing["cues"][0]["start"]
        assert b_wav > 0.8  # 是 B（念三遍，0.88 秒起）的音频，不是 A 的（0.56 秒）
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert {r["voice"] for r in manifest["beats"].values()} == {b_voice} and manifest["mixed"] is False
    assert manifest["policy_ok"] is (b_voice == "Kore")
    capsys.readouterr()
    rerun = ["synth", str(scene), "--out", str(out)] + (["--voice", "Aoede"] if b_voice == "Aoede" else [])
    assert main(rerun) == 0  # 最后一套原样沿用，不再调用 TTS
    printed = capsys.readouterr().out
    assert len(speak.calls) == 8 and printed.count("沿用") == 2


def test_no_wait_fails_fast_when_the_audio_dir_is_busy(tmp_path, monkeypatch, capsys):
    _clean_env(monkeypatch)
    speak = Speaker()
    monkeypatch.setattr(synth, "default_speak", speak)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    out.mkdir()
    with ttscache.dir_lock(out):  # 另一个 synth 正在跑
        started = time.monotonic()
        assert main(["synth", str(scene), "--out", str(out), "--no-wait"]) == 2
        assert main(["align", str(scene), "--audio", str(out), "--no-wait"]) == 2
        assert main(["pitch", str(scene), "--audio", str(out), "--fix", "--no-wait"]) == 2
        assert time.monotonic() - started < 5
    err = capsys.readouterr().err
    assert err.count("正被另一个 synth") == 3 and "--no-wait" in err
    assert speak.calls == [] and not (out / "manifest.json").exists()


def test_refreshing_a_key_never_changes_audio_a_reader_already_holds(tmp_path):
    """--no-cache / pitch --fix 刷新同一个键：只写新对象、换索引；拿着旧路径的读者读到的仍是完整的旧音频。"""
    cache = ttscache.tts_dir(tmp_path)
    cache.mkdir(parents=True)

    def produce(text):
        tmp = ttscache.scratch(cache, "k")
        synth._to_wav(voice_like(text), tmp)
        return tmp

    old = ttscache.store(cache, "k", produce("旧的"), synth.RATE, {"kind": "tts"})
    held = ttscache.lookup(cache, "k", synth.RATE)[0]
    before, stat = held.read_bytes(), held.stat()
    new = ttscache.store(cache, "k", produce("新的一版，长一些"), synth.RATE, {"kind": "tts"})
    assert held == old != new and held.read_bytes() == before  # 读者手里的文件没变
    assert ttscache.lookup(cache, "k", synth.RATE)[0] == new
    again = ttscache.store(cache, "k2", produce("新的一版，长一些"), synth.RATE, {"kind": "tts"})
    assert again == new and new.stat().st_ino == ttscache.lookup(cache, "k", synth.RATE)[0].stat().st_ino
    same = ttscache.put_object(cache, produce("旧的"), synth.RATE)[0]  # 同样内容再写一次：已有的对象原样不动
    assert same == old and (same.stat().st_ino, same.stat().st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
    assert not [p for p in tmp_path.rglob("*") if ".tmp" in p.name]


def test_speed_variant_is_keyed_by_the_bytes_ffmpeg_consumed(tmp_path, monkeypatch):
    import hashlib
    import subprocess

    cache = ttscache.tts_dir(tmp_path)
    cache.mkdir(parents=True)
    tmp = ttscache.scratch(cache, "raw")
    synth._to_wav(voice_like("一句话"), tmp)
    raw = ttscache.store(cache, "raw", tmp, synth.RATE, {"kind": "tts"})
    fed, real_run = [], subprocess.run

    def run_spy(cmd, **kw):
        if "atempo=1.25" in " ".join(cmd):
            fed.append(kw.get("input"))
        return real_run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", run_spy)
    fast = synth._speed(raw, 1.25, cache)
    assert len(fed) == 1 and fed[0] == raw.read_bytes()  # ffmpeg 从管道读的就是算键的那份字节
    index = next(json.loads(p.read_text()) for p in cache.glob("*.json") if json.loads(p.read_text())["kind"] == "speed")
    assert index["source_sha256"] == hashlib.sha256(fed[0]).hexdigest() and index["sha256"] == fast.stem
    assert synth._speed(raw, 1.25, cache) == fast and len(fed) == 1  # 第二次命中


def test_manifest_matches_published_beats_after_an_interrupted_run(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    monkeypatch.setattr(synth, "default_speak", Speaker())
    assert main(["synth", str(scene), "--out", str(out)]) == 0

    def dies_on_next(instructions, text, model):
        if text == "下一段。":
            raise RuntimeError("断网")
        return voice_like(text * 2)

    monkeypatch.setattr(synth, "default_speak", dies_on_next)
    with pytest.raises(RuntimeError):
        main(["synth", str(scene), "--out", str(out), "--voice", "Aoede"])
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["beats"]["Hook"]["voice"] == "Aoede"  # Hook 已经用 Aoede 发布了，manifest 也这么记
    assert manifest["beats"]["Next"]["voice"] == "Kore" and manifest["mixed"] is True and manifest["policy_ok"] is False
    _consistent(out, "Hook")
    with ttscache.dir_lock(out, wait=False):  # 异常退出也放了锁
        pass


@pytest.mark.parametrize("stage", ["wav", "timing", "stamp"])
def test_interrupted_publication_is_reassembled_next_time(tmp_path, monkeypatch, stage):
    """发布 wav、timing.json、说明之间任何一步被打断：下次不会拿半套旧文件“沿用”，而是重拼成一致的一套。"""
    speak = Speaker()
    run(beat_of(SAY), tmp_path, speak)
    _align(tmp_path)
    edited = [dict(s) for s in SAY]
    edited[2]["text"] = "第三句改长了一些？"
    real_replace, real_write = os.replace, ttscache.write_text_atomic
    target = {"wav": tmp_path / "Hook.wav", "timing": tmp_path / "Hook.timing.json",
              "stamp": ttscache.beats_dir(tmp_path) / "Hook.json"}[stage]

    def replace(src, dst):
        if Path(dst) == target:
            raise KeyboardInterrupt("断电")
        return real_replace(src, dst)

    def write(path, text):
        if Path(path) == target:
            raise KeyboardInterrupt("断电")
        return real_write(path, text)

    with monkeypatch.context() as m:
        m.setattr(os, "replace", replace)
        m.setattr(ttscache, "write_text_atomic", write)
        with pytest.raises(KeyboardInterrupt):
            run(beat_of(edited), tmp_path, speak)
    stamp = json.loads((ttscache.beats_dir(tmp_path) / "Hook.json").read_text())
    if stage != "wav":  # wav 已经换了：旧说明的 sha256 对不上，不会被拿来“沿用”
        assert stamp["wav_sha256"] != ttscache.sha256_file(tmp_path / "Hook.wav")

    logs = []
    timing = run(beat_of(edited), tmp_path, speak, logs)
    assert len(speak.calls) == 4  # 改的那句在被打断那次已经进了缓存
    assert "沿用" not in logs[-1] and logs[-1].startswith("Hook: 3 cached, 0 synthesized")
    assert _consistent(tmp_path) == timing and "marks" not in timing
    logs.clear()
    run(beat_of(edited), tmp_path, speak, logs)
    assert "沿用" in logs[-1]


def test_lock_held_by_another_process_blocks_synth(tmp_path, monkeypatch, capsys):
    """真正的另一个进程拿着锁（flock 是跨进程的）：--no-wait 立刻失败；默认等它放锁再跑。"""
    import subprocess
    import sys

    _clean_env(monkeypatch)
    speak = Speaker()
    monkeypatch.setattr(synth, "default_speak", speak)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    out.mkdir()
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys; from pathlib import Path; from voice import ttscache\n"
         f"with ttscache.dir_lock(Path({str(out)!r})):\n    print('held', flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert main(["synth", str(scene), "--out", str(out), "--no-wait"]) == 2
        timer = threading.Timer(0.5, holder.stdin.close)  # 半秒后对方放锁
        timer.start()
        started = time.monotonic()
        assert main(["synth", str(scene), "--out", str(out)]) == 0
        assert time.monotonic() - started >= 0.4 and len(speak.calls) == 4
        assert "waiting for another synth on" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()
