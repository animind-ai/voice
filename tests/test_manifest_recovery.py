"""manifest 和音频的一致性：发布途中在任何一步被打断，manifest 都不能把不合规的音频说成合规；
下一次运行（哪怕只合成别的 beat）按发布说明把记录修好。全部用假的 speak，不联网。"""

import json
import math
import os
import struct

import pytest

from voice import policy, synth, ttscache
from voice.cli import main

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
      - {id: s2, pace: ask, text: "第二句？"}
  - id: Next
    say:
      - {pace: normal, text: "下一段。"}
"""


class Crash(BaseException):
    """模拟进程被杀：不是 Exception，谁也不会把它吞掉。"""


def speak(instructions, text, model):
    """声音不同，音频不同（Aoede 念得长一倍），这样才能从 wav 认出是谁念的。"""
    if os.environ.get("GEMINI_TTS_VOICE") == "Aoede":
        text = text * 2
    rate, freq = 24000, 200 + sum(map(ord, text)) % 30
    n = int((0.4 + 0.04 * len(text)) * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * k / rate))) for k in range(n))


@pytest.fixture
def episode(tmp_path, monkeypatch):
    for name in (policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(synth, "default_speak", speak)
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0  # 先全部用合规的 Kore 合成
    kore = {b: ttscache.sha256_file(out / f"{b}.wav") for b in ("Hook", "Next")}
    return scene, out, kore


def manifest(out):
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


def assert_never_overclaims(out, kore):
    """manifest 说合规的，wav 必须真是 Kore 念的那一版。"""
    data = manifest(out)
    for beat_id, record in data["beats"].items():
        if record.get("policy_ok"):
            assert ttscache.sha256_file(out / f"{beat_id}.wav") == kore[beat_id], (beat_id, record)
    if data["policy_ok"]:
        assert all(ttscache.sha256_file(out / f"{b}.wav") == d for b, d in kore.items())


def crash_at(monkeypatch, out, boundary):
    """在发布 Hook 的某一步让进程“死掉”。"""
    real_replace, real_write = os.replace, ttscache.write_text_atomic
    hook_wav, hook_timing = out / "Hook.wav", out / "Hook.timing.json"
    hook_stamp, manifest_path = ttscache.beats_dir(out) / "Hook.json", out / "manifest.json"
    manifest_writes = []

    def replace(src, dst):
        if boundary == "before_wav" and os.fspath(dst) == os.fspath(hook_wav):
            raise Crash
        return real_replace(src, dst)

    def write(path, text):
        if boundary == "after_wav" and path == hook_timing:
            raise Crash
        if boundary == "after_timing" and path == hook_stamp:
            raise Crash
        if path == manifest_path:
            manifest_writes.append(text)
            # 预写那一次（第 1 次）失败，或提交那一次（第 2 次）写到一半：留下半个临时文件
            if (boundary == "pending_write" and len(manifest_writes) == 1) or \
                    (boundary == "manifest_write" and len(manifest_writes) == 2):
                ttscache.temp_path(path).write_text(text[: len(text) // 2], encoding="utf-8")
                raise Crash
        return real_write(path, text)

    def commit(*args, **kwargs):
        raise Crash

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(ttscache, "write_text_atomic", write)
    monkeypatch.setattr(policy, "write_text_atomic", write)  # policy 按名字导入了它
    if boundary == "after_stamp":
        monkeypatch.setattr(policy, "write_manifest", commit)


# 被打断之后 Hook.wav 里是谁的声音：wav 还没换就还是 Kore，换了就是 Aoede
BOUNDARIES = {
    "pending_write": "Kore",   # 预写 manifest 时就断了：什么都还没换
    "before_wav": "Kore",      # 预写完、换 wav 之前
    "after_wav": "Aoede",      # 换了 wav、还没写 timing.json
    "after_timing": "Aoede",   # 写了 timing.json、还没写说明
    "after_stamp": "Aoede",    # 写了说明、还没提交 manifest
    "manifest_write": "Aoede",  # 提交 manifest 写到一半
}


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_interrupted_publication_never_overclaims_and_self_heals(episode, monkeypatch, capsys, boundary):
    scene, out, kore = episode
    with monkeypatch.context() as m:
        crash_at(m, out, boundary)
        with pytest.raises(Crash):
            main(["synth", str(scene), "--out", str(out), "--voice", "Aoede", "--only", "Hook"])
    in_wav = "Kore" if ttscache.sha256_file(out / "Hook.wav") == kore["Hook"] else "Aoede"
    assert in_wav == BOUNDARIES[boundary]
    assert_never_overclaims(out, kore)
    if in_wav == "Aoede":
        assert manifest(out)["policy_ok"] is False  # Aoede 的音频已经发布了：绝不能还说合规

    # 下一次运行只合成另一个 beat（Next 没变，直接沿用）：Hook 的记录也要修好
    capsys.readouterr()
    assert main(["synth", str(scene), "--out", str(out), "--only", "Next"]) == 0
    data = manifest(out)
    hook = data["beats"]["Hook"]
    assert hook == {"model": "google/gemini-3.8-flash-tts", "voice": in_wav, "speed": 1.0,
                    "policy_ok": in_wav == "Kore"}  # 按说明恢复成 wav 里真正的来源，没有 pending / unknown
    assert data["policy_ok"] is (in_wav == "Kore") and data["mixed"] is (in_wav == "Aoede")
    assert_never_overclaims(out, kore)
    err = capsys.readouterr().err
    assert ("manifest 按发布说明修正：Hook" in err) is (boundary not in ("pending_write",))

    # 再把 Hook 用 Kore 重合成：整个目录回到合规
    assert main(["synth", str(scene), "--out", str(out), "--only", "Hook"]) == 0
    assert manifest(out)["policy_ok"] is True and manifest(out)["mixed"] is False


def test_wav_replaced_outside_synth_is_marked_unknown(episode):
    scene, out, kore = episode
    (out / "Hook.wav").write_bytes((out / "Next.wav").read_bytes())  # 有人手动换了音频：哪份说明都对不上
    assert main(["synth", str(scene), "--out", str(out), "--only", "Next"]) == 0
    data = manifest(out)
    assert data["beats"]["Hook"]["unknown"] is True and data["policy_ok"] is False


def test_unchanged_audio_with_another_voice_is_republished_with_its_provenance(tmp_path, monkeypatch):
    """假 speak 不管声音，两次的音频字节完全一样；来源在指纹里，所以照样重新发布，说明和 manifest 记的是新声音。"""
    for name in (policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(synth, "default_speak", lambda instructions, text, model: _same(text))
    scene = tmp_path / "t.scene.yaml"
    scene.write_text(SCENE, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0
    assert main(["synth", str(scene), "--out", str(out), "--voice", "Aoede"]) == 0
    stamp = json.loads((ttscache.beats_dir(out) / "Hook.json").read_text())
    assert stamp["voice"] == "Aoede" and manifest(out)["beats"]["Hook"]["voice"] == "Aoede"
    assert policy.reconcile(out) == []  # manifest 和说明一致，没什么可修


def _same(text):
    rate = 24000
    n = int((0.4 + 0.04 * len(text)) * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 210 * k / rate))) for k in range(n))


def test_speed_key_keeps_full_float_precision(tmp_path):
    cache = ttscache.tts_dir(tmp_path)
    cache.mkdir(parents=True)
    tmp = ttscache.scratch(cache, "raw")
    synth._to_wav(_same("一句话"), tmp)
    raw = ttscache.store(cache, "raw", tmp, synth.RATE, {"kind": "tts"})
    synth._speed(raw, 1.250001, cache)
    synth._speed(raw, 1.250002, cache)
    synth._speed(raw, 1.250001, cache)  # 命中，不再新增
    speeds = sorted(json.loads(p.read_text())["speed"] for p in cache.glob("*.json")
                    if json.loads(p.read_text())["kind"] == "speed")
    assert speeds == ["1.250001", "1.250002"]
