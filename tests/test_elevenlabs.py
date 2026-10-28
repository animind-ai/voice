"""ElevenLabs 提供方：模型选择、环境变量、SDK 调用参数、按字数记用量，以及升级前的句子缓存照样命中。

SDK 用假模块代替，不联网，不需要 ELEVENLABS_API_KEY。
"""

import json
import math
import os
import struct
import sys
import types

import pytest
from scene_spec import silence_after

from voice import policy, script, synth
from voice.cli import main

EL_ENV = ("TTS_PROVIDER", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL_ID", "ELEVENLABS_STABILITY",
          "ELEVENLABS_SIMILARITY_BOOST", "ELEVENLABS_USD_PER_CHAR", "ELEVENLABS_API_KEY")


def pcm(seconds: float, freq: float = 140.0, rate: int = 24000) -> bytes:
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * k / rate))) for k in range(n))


@pytest.fixture
def env(monkeypatch):
    """清掉所有相关环境变量；monkeypatch 会在测试后还原 cli 写进 os.environ 的声音。"""
    for name in (*EL_ENV, policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return monkeypatch


# ---------- 模型和声音从哪来 ----------

def test_default_model_follows_tts_provider():
    assert synth.default_model({}) is None
    assert synth.default_model({"GEMINI_TTS_MODEL": "google/gemini-3.8-flash-tts"}) == "google/gemini-3.8-flash-tts"
    assert synth.default_model({"TTS_PROVIDER": "elevenlabs", "GEMINI_TTS_MODEL": "g"}) == "elevenlabs/eleven_v4"
    assert synth.default_model({"TTS_PROVIDER": "ElevenLabs", "ELEVENLABS_MODEL_ID": "eleven_v5"}) == "elevenlabs/eleven_v5"
    assert synth.default_model({"TTS_PROVIDER": "gemini", "GEMINI_TTS_MODEL": "g"}) == "g"


def test_default_voice_and_voice_env_by_model():
    environ = {"ELEVENLABS_VOICE_ID": "el-id", "GEMINI_TTS_VOICE": "Kore"}
    assert synth.voice_env("elevenlabs/eleven_v4") == "ELEVENLABS_VOICE_ID"
    assert synth.voice_env("google/gemini-3.8-flash-tts") == "GEMINI_TTS_VOICE"
    assert synth.default_voice("elevenlabs/eleven_v4", environ) == "el-id"
    assert synth.default_voice("google/gemini-3.8-flash-tts", environ) == "Kore"
    assert synth.default_voice(None, environ) == "Kore"


def test_policy_tts_provider_is_an_explicit_choice():
    doc = {"style": {"voice": {"persona": "teacher-hs", "speed": 1.0}}}
    environ = {"TTS_PROVIDER": "elevenlabs", "ELEVENLABS_VOICE_ID": "el-id", "ELEVENLABS_MODEL_ID": "eleven_v4",
               "GEMINI_TTS_VOICE": "Puck"}
    choice = policy.resolve(doc, environ=environ)  # 不需要 --allow-env-voice
    assert (choice.model, choice.voice, choice.speed) == ("elevenlabs/eleven_v4", "el-id", 1.0)
    assert choice.sources == {"model": "env", "voice": "env", "speed": "scene"}
    assert len(choice.warnings) == 2 and not choice.policy_ok
    # 场景里写死的 Gemini 声音也让位于显式选的提供方
    scene = {"style": {"voice": {"model": "google/gemini-3.8-flash-tts", "voice": "Kore"}}}
    assert policy.resolve(scene, environ=environ).voice == "el-id"
    # 命令行仍然最高
    assert policy.resolve(doc, voice="other-id", environ=environ).voice == "other-id"


def test_policy_elevenlabs_needs_a_voice_id():
    with pytest.raises(ValueError, match="ELEVENLABS_VOICE_ID"):
        policy.resolve({}, environ={"TTS_PROVIDER": "elevenlabs"})
    # --model elevenlabs/… 不设 TTS_PROVIDER：声音也从 ELEVENLABS_VOICE_ID 取，而不是默认的 Kore
    choice = policy.resolve({}, model="elevenlabs/eleven_v4", environ={"ELEVENLABS_VOICE_ID": "el-id"})
    assert (choice.model, choice.voice) == ("elevenlabs/eleven_v4", "el-id")


def test_policy_without_provider_is_unchanged():
    choice = policy.resolve({}, environ={"ELEVENLABS_VOICE_ID": "el-id", "GEMINI_TTS_VOICE": "Puck"})
    assert (choice.model, choice.voice) == ("google/gemini-3.8-flash-tts", "Kore") and choice.policy_ok


# ---------- SDK 调用 ----------

class FakeSDK:
    def __init__(self, audio: bytes):
        self.audio, self.calls, self.clients = audio, [], []

    def install(self, monkeypatch):
        sdk = self

        class VoiceSettings:
            def __init__(self, **kw):
                self.kw = kw

        class Raw:
            def convert(self, **kw):
                sdk.calls.append(kw)
                chunks = [sdk.audio[:10], sdk.audio[10:]] if sdk.audio else []

                class Response:
                    def __enter__(self):
                        return types.SimpleNamespace(data=iter(chunks))

                    def __exit__(self, *exc):
                        return False

                return Response()

        class ElevenLabs:
            def __init__(self, **kw):
                sdk.clients.append(kw)
                self.text_to_speech = types.SimpleNamespace(with_raw_response=Raw())

        monkeypatch.setitem(sys.modules, "elevenlabs", types.SimpleNamespace(ElevenLabs=ElevenLabs, VoiceSettings=VoiceSettings))
        return self


def test_elevenlabs_speak_sends_env_settings(env):
    sdk = FakeSDK(b"\x01\x02" * 50).install(env)
    env.setenv("ELEVENLABS_VOICE_ID", "el-id")
    env.setenv("ELEVENLABS_STABILITY", "0.4")
    env.setenv("ELEVENLABS_SIMILARITY_BOOST", "0.9")
    assert synth.default_speak("人设", "一句中文", "elevenlabs/eleven_v4") == b"\x01\x02" * 50
    call = sdk.calls[0]
    assert {k: call[k] for k in ("text", "voice_id", "model_id", "language_code", "output_format")} == {
        "text": "一句中文", "voice_id": "el-id", "model_id": "eleven_v4", "language_code": "zh", "output_format": "pcm_24000"}
    assert call["voice_settings"].kw == {"stability": 0.4, "similarity_boost": 0.9}
    assert "api_key" not in sdk.clients[0]  # 密钥由 SDK 自己从 ELEVENLABS_API_KEY 读
    synth.elevenlabs_speak("", "An English line.", "elevenlabs/eleven_v4")
    assert sdk.calls[1]["language_code"] == "en"
    assert sdk.calls[1]["voice_settings"].kw == {"stability": 0.4, "similarity_boost": 0.9}


def test_elevenlabs_speak_defaults_and_empty_audio(env):
    sdk = FakeSDK(b"").install(env)
    env.setenv("ELEVENLABS_VOICE_ID", "el-id")
    with pytest.raises(RuntimeError, match="没有返回音频"):
        synth.elevenlabs_speak("", "一句", "elevenlabs/eleven_v4")
    assert sdk.calls[0]["voice_settings"].kw == {"stability": 0.5, "similarity_boost": 0.75}


# ---------- 用量 ----------

def test_usage_logged_per_char_without_price(tmp_path, env):
    log = tmp_path / "tts_usage.jsonl"
    assert synth.log_usage(log, "B-0", "elevenlabs/eleven_v4", "一二三四五", "人设", 1.2) is None
    synth.log_usage(log, "B-1", "google/gemini-3.8-flash-tts", "一二三", "人设", 1.0)
    env.setenv("ELEVENLABS_USD_PER_CHAR", "0.0001")
    assert synth.log_usage(log, "B-2", "elevenlabs/eleven_v4", "一二", "人设", 0.5) == pytest.approx(0.0002)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["est_usd"] is None and rows[0]["chars"] == 5 and rows[0]["model"] == "elevenlabs/eleven_v4"
    assert rows[1]["est_usd"] > 0 and rows[2]["est_usd"] == pytest.approx(0.0002)
    calls, usd = synth.total_cost(log)
    assert calls == 3 and usd == pytest.approx(rows[1]["est_usd"] + 0.0002)
    assert synth.total_chars(log) == 5  # 只算单价未知的


# ---------- cli 端到端 ----------

SCENE = """meta:
  id: el-01
style:
  voice:
    persona: teacher-hs
    speed: 1.0
beats:
  - id: B
    say:
      - {pace: normal, text: "第一句。"}
      - {pace: normal, text: "第二句，"}
      - {pace: ask, text: "第三句？"}
"""


def test_cli_synth_with_elevenlabs(tmp_path, env, capsys):
    spec = tmp_path / "el.scene.yaml"
    spec.write_text(SCENE, encoding="utf-8")
    keys = tmp_path / "el.env"  # 只有非密钥的值；没有 API key 也不会用到（SDK 是假的）
    keys.write_text("TTS_PROVIDER=elevenlabs\nELEVENLABS_VOICE_ID=el-id\nELEVENLABS_MODEL_ID=eleven_v4\n", encoding="utf-8")
    for name in ("TTS_PROVIDER", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL_ID"):
        env.setenv(name, "")  # load_env 用 setdefault：先登记再删，测试后 monkeypatch 会还原
        env.delenv(name)
    calls = []

    def speak(instructions, text, model):
        calls.append((text, model, os.environ.get("ELEVENLABS_VOICE_ID")))
        return pcm(0.6)

    env.setattr(synth, "elevenlabs_speak", speak)
    out = tmp_path / "audio"
    assert main(["synth", str(spec), "--out", str(out), "--env", str(keys)]) == 0
    printed = capsys.readouterr()
    assert [c[1:] for c in calls] == [("elevenlabs/eleven_v4", "el-id")] * 3
    assert "这次 12 字" in printed.out and "这次调用 TTS 3 次" in printed.out
    rows = [json.loads(line) for line in (out / "tts_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and all(r["est_usd"] is None for r in rows)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert json.dumps(manifest).count("el-id") >= 1 and "elevenlabs/eleven_v4" in json.dumps(manifest)
    # 句间静音按标点（silence_after），不是按语气
    cues = json.loads((out / "B.timing.json").read_text(encoding="utf-8"))["cues"]
    gaps = [cues[i + 1]["start"] - cues[i]["end"] for i in range(len(cues) - 1)]
    assert gaps == pytest.approx([silence_after("第一句。"), silence_after("第二句，")], abs=2e-3)
    # 再跑一次：全部命中缓存
    calls.clear()
    assert main(["synth", str(spec), "--out", str(out), "--env", str(keys)]) == 0
    assert calls == []


def test_branch_era_lines_are_cache_hits(tmp_path, env):
    """elevenlabs-provider 分支合成的目录（只有 lines/<beat>-<i>.wav + 说明，没有 .cache/）：
    说明里的 text / instructions / voice / model 对得上就收进缓存，不调用 TTS，并按标点重拼。"""
    doc = {"beats": [{"id": "B", "say": [{"pace": "normal", "text": "第一句。"}, {"pace": "ask", "text": "第二句？"}]}]}
    beat = script.beats(doc)[0]
    lines = tmp_path / "lines"
    lines.mkdir()
    model, voice = "elevenlabs/eleven_v4", "el-id"
    for s in beat.sentences:  # 分支的写法：lines/<name>.wav + {"text", "instructions", "voice", "model"}
        synth._to_wav(pcm(0.7), lines / f"B-{s.index}.wav")
        stamp = {"text": s.spoken, "instructions": synth.instructions_for("teacher-hs", s.pace), "voice": voice, "model": model}
        (lines / f"B-{s.index}.json").write_text(json.dumps(stamp, ensure_ascii=False), encoding="utf-8")

    def no_tts(*_):
        raise AssertionError("不该调用 TTS")

    timing = synth.synth_beat(beat, tmp_path, "teacher-hs", model, voice, speak=no_tts, log=lambda *_: None)
    gap = timing["cues"][1]["start"] - timing["cues"][0]["end"]
    assert gap == pytest.approx(silence_after("第一句。"), abs=2e-3)
    assert not (tmp_path / "tts_usage.jsonl").exists()
    assert len(list((tmp_path / ".cache" / "tts").glob("*.json"))) == 2
