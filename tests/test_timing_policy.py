"""句间静音（silence_after + say[].hold）、时间报告（scene_spec.check_timing）、声音取值policy 和 manifest。

全部用假的 speak，不联网；音高用 210 Hz（女声 Kore 的范围内），免得触发重合成。
"""

import json
import math
import os
import shutil
import struct

import pytest
from scene_spec import silence_after
from scene_spec.pauses import GAP_TOLERANCE
from scene_spec.quality import check_timing, check_voice
from scene_spec.report import Report

from voice import policy, script, synth, timing_report
from voice.cli import main


def voice_like(seconds: float, freq: float = 210.0, rate: int = 24000) -> bytes:
    n = int(seconds * rate)
    return b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * k / rate))) for k in range(n))


def fake_speak(calls=None):
    def speak(instructions, text, model):
        if calls is not None:
            calls.append({"text": text, "model": model, "voice": os.environ.get("GEMINI_TTS_VOICE")})
        return voice_like(0.5 + 0.03 * len(text))

    return speak


POLICY_VOICE = {"model": "google/gemini-3.8-flash-tts", "voice": "Kore", "speed": 1.0}


def beat_doc(say, voice=POLICY_VOICE):
    doc = {"beats": [{"id": "B", "say": say}]}
    if voice is not None:
        doc["style"] = {"voice": voice}
    return doc


SAY = [
    {"pace": "normal", "text": "第一句。"},              # 句号，默认 0.45
    {"pace": "normal", "text": "第二句，", "hold": 0.3},  # 逗号，hold 覆盖默认 0.22
    {"pace": "ask", "text": "第三句？"},                  # 问号，默认 0.60
    {"pace": "normal", "text": "最后一句。"},
]


def gaps_of(cues):
    return [cues[i + 1]["start"] - cues[i]["end"] for i in range(len(cues) - 1)]


@pytest.mark.parametrize("speed", [1.0, 1.25])
def test_gaps_follow_silence_after_including_hold(tmp_path, speed):
    doc = beat_doc(SAY)
    beat = script.beats(doc)[0]
    timing = synth.synth_beat(beat, tmp_path, "teacher-hs", "m", "Kore", speed, speak=fake_speak(), log=lambda *_: None)
    expected = [silence_after(s["text"], s.get("hold")) for s in SAY[:-1]]
    assert expected == pytest.approx([0.45, 0.3, 0.60])
    for got, want in zip(gaps_of(timing["cues"]), expected):
        assert abs(got - want) <= GAP_TOLERANCE / 10  # 毫秒圆整之内，远小于 SS487 容差


def test_hold_out_of_range_or_nonfinite_is_rejected():
    for bad in (1.0, 0.1, float("nan"), float("inf"), True, "0.3", None):
        doc = beat_doc([{"pace": "normal", "text": "第一句。", "hold": bad}])
        with pytest.raises(ValueError, match="hold"):
            script.beats(doc)
    # 句号的区间是 0.25–0.8，端点含在内
    assert script.beats(beat_doc([{"pace": "normal", "text": "第一句。", "hold": 0.8}]))[0].sentences[0].hold == 0.8


def test_timing_report_passes_check_timing_and_check_voice(tmp_path):
    doc = beat_doc(SAY)
    beat = script.beats(doc)[0]
    synth.synth_beat(beat, tmp_path, "teacher-hs", "m", "Kore", 1.0, speak=fake_speak(), log=lambda *_: None)
    report_dict = timing_report.build([beat], tmp_path)
    assert set(report_dict["beats"]["B"]) == {"cues"} and len(report_dict["beats"]["B"]["cues"]) == len(SAY)

    report = Report()
    check_voice(doc, report)
    check_timing(doc, report_dict, report)
    assert report.ok, [str(i) for i in report.issues]
    assert not {"SS482", "SS486", "SS487"} & set(report.codes())

    # 把一处静音改成 0.9 秒，SS487 必须报出来（确认检查真的在看 cues）
    tampered = json.loads(json.dumps(report_dict))
    tampered["beats"]["B"]["cues"][1]["start"] += 0.6
    for c in tampered["beats"]["B"]["cues"][2:]:
        c["start"] += 0.6
        c["end"] += 0.6
    bad = Report()
    check_timing(doc, tampered, bad)
    assert "SS487" in bad.codes()


# ---------- cli：声音取值、manifest、时间报告 ----------

SCENE = """meta:
  id: t-01
style:
  voice:
    persona: teacher-hs
{voice}beats:
  - id: B
    say:
      - {{pace: normal, text: "第一句。"}}
      - {{pace: normal, text: "第二句，", hold: 0.3}}
      - {{pace: ask, text: "第三句？"}}
"""


def write_scene(tmp_path, voice_lines: str | None):
    body = "" if voice_lines is None else "".join(f"    {line}\n" for line in voice_lines.splitlines())
    path = tmp_path / "t.scene.yaml"
    path.write_text(SCENE.format(voice=body), encoding="utf-8")
    return path


@pytest.fixture
def clean_env(monkeypatch):
    for name in (policy.ENV_MODEL, policy.ENV_VOICE, policy.ENV_ALLOW):
        monkeypatch.setenv(name, "")  # monkeypatch 会在测试后还原 cli 对 GEMINI_TTS_VOICE 的赋值
    return monkeypatch


def test_cli_scene_voice_is_source_of_truth_and_env_is_ignored(tmp_path, clean_env, capsys):
    clean_env.setenv(policy.ENV_MODEL, "other/tts-model")
    clean_env.setenv(policy.ENV_VOICE, "Puck")
    calls = []
    clean_env.setattr(synth, "default_speak", fake_speak(calls))
    scene = write_scene(tmp_path, "voice: Kore\nmodel: google/gemini-3.8-flash-tts\nspeed: 1.0")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["model"], manifest["voice"], manifest["speed"]) == ("google/gemini-3.8-flash-tts", "Kore", 1.0)
    assert manifest["policy_ok"] is True and manifest["sources"]["voice"] == "scene"
    assert {c["model"] for c in calls} == {"google/gemini-3.8-flash-tts"}
    assert {c["voice"] for c in calls} == {"Kore"}
    err = capsys.readouterr().err
    assert "未采用" in err and "WARNING" not in err and "⚠⚠" not in err
    # check_timing 能直接吃 cli 写出的时间报告
    report = Report()
    check_timing(script_doc(scene), json.loads((out / timing_report.REPORT_NAME).read_text(encoding="utf-8")), report)
    assert report.ok, [str(i) for i in report.issues]


def script_doc(scene):
    from scene_spec.loader import load_file

    return load_file(scene)


def test_cli_defaults_when_scene_has_no_style_voice(tmp_path, clean_env):
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = write_scene(tmp_path, None)
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["model"], manifest["voice"], manifest["speed"]) == ("google/gemini-3.8-flash-tts", "Kore", 1.0)
    assert manifest["policy_ok"] is True and manifest["sources"]["voice"] == "default"


def test_env_overrides_only_with_allow_flag_and_loud_warning(tmp_path, clean_env, capsys):
    clean_env.setenv(policy.ENV_VOICE, "Puck")
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = write_scene(tmp_path, "voice: Kore\nmodel: google/gemini-3.8-flash-tts\nspeed: 1.0")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out), "--allow-env-voice"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["voice"] == "Puck" and manifest["sources"]["voice"] == "env" and manifest["policy_ok"] is False
    err = capsys.readouterr().err
    assert "⚠⚠" in err and "GEMINI_TTS_VOICE=Puck" in err and "WARNING" in err and "不可发布" in err

    # 同样的效果也可以用环境变量开关
    out2 = tmp_path / "audio2"
    clean_env.setenv(policy.ENV_ALLOW, "1")
    assert main(["synth", str(scene), "--out", str(out2)]) == 0
    assert json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))["voice"] == "Puck"
    assert "⚠⚠" in capsys.readouterr().err


def test_cli_flag_overrides_scene_with_notice_and_policy_warning(tmp_path, clean_env, capsys):
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = write_scene(tmp_path, "voice: Kore\nmodel: google/gemini-3.8-flash-tts\nspeed: 1.0")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out), "--voice", "Aoede", "--speed", "1.1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["voice"], manifest["speed"], manifest["sources"]["voice"]) == ("Aoede", 1.1, "flag")
    err = capsys.readouterr().err
    assert "覆盖" in err and "WARNING" in err


def test_nonfinite_or_nonpositive_speed_is_refused():
    for bad in (float("nan"), float("inf"), 0, -1.0):
        with pytest.raises(ValueError, match="speed|语速"):
            policy.resolve({"style": {"voice": {"speed": bad}}}, environ={})


TWO_BEATS = SCENE.format(voice="") + '  - id: C\n    say:\n      - {pace: normal, text: "另一集。"}\n'


def test_full_rerun_with_another_voice_is_consistent_not_mixed(tmp_path, clean_env, capsys):
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = tmp_path / "two.scene.yaml"
    scene.write_text(TWO_BEATS, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0
    assert main(["synth", str(scene), "--out", str(out), "--voice", "Puck"]) == 0
    assert "不同的声音" not in capsys.readouterr().err  # 两个 beat 都换成了 Puck：目录一致，只是不合规（另有政策警告）
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mixed"] is False and manifest["voice"] == "Puck" and manifest["policy_ok"] is False
    assert set(manifest["beats"]) == {"B", "C"}


def test_partial_resynth_marks_directory_mixed_and_persists(tmp_path, clean_env, capsys):
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = tmp_path / "two.scene.yaml"
    scene.write_text(TWO_BEATS, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0  # 两个 beat 都是 Kore，合规
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["policy_ok"] is True

    # 只重合成 C 为 Puck：B 仍是旧的 Kore 音频，目录混了两种声音，必须不合规、必须提醒
    assert main(["synth", str(scene), "--out", str(out), "--only", "C", "--voice", "Puck"]) == 0
    assert "不同的声音" in capsys.readouterr().err
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["beats"]["B"]["voice"] == "Kore" and manifest["beats"]["C"]["voice"] == "Puck"
    assert manifest["mixed"] is True and manifest["policy_ok"] is False
    assert (manifest["model"], manifest["voice"], manifest["speed"]) == (None, None, None)

    # 同样的命令再跑一次：混用还在，警告不能消失
    assert main(["synth", str(scene), "--out", str(out), "--only", "C", "--voice", "Puck"]) == 0
    assert "不同的声音" in capsys.readouterr().err
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["mixed"] is True


def test_pitch_fix_uses_each_beats_recorded_speed(tmp_path, clean_env):
    scene = tmp_path / "two.scene.yaml"
    scene.write_text(TWO_BEATS, encoding="utf-8")
    out = tmp_path / "audio"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"beats": {"B": {"speed": 1.2}}}), encoding="utf-8")
    assert policy.beat_speed(out, "B", 1.0) == 1.2
    assert policy.beat_speed(out, "C", 1.0) == 1.0  # 没有记录的 beat 用场景里的语速


def test_policy_resolve_defaults_and_sources():
    choice = policy.resolve({}, environ={})
    assert (choice.model, choice.voice, choice.speed) == ("google/gemini-3.8-flash-tts", "Kore", 1.0)
    assert choice.policy_ok and choice.sources == {"model": "default", "voice": "default", "speed": "default"}
    choice = policy.resolve({"style": {"voice": {"voice": "Puck"}}}, environ={policy.ENV_VOICE: "Aoede"})
    assert choice.voice == "Puck" and choice.sources["voice"] == "scene" and not choice.policy_ok
    assert any("未采用" in n for n in choice.notices)


def test_legacy_dir_without_manifest_is_not_declared_compliant(tmp_path, clean_env, capsys):
    # 升级前生成的目录：B 是 Puck，manifest 丢了。只重合成 C（Kore）不能把整个目录标成合规
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = tmp_path / "two.scene.yaml"
    scene.write_text(TWO_BEATS, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out), "--voice", "Puck"]) == 0
    (out / "manifest.json").unlink()  # 模拟升级前的目录：音频还在，来源记录没有
    shutil.rmtree(out / ".cache")  # 升级前也没有发布说明（有说明时 manifest 会按说明恢复，见 test_sentence_cache）

    assert main(["synth", str(scene), "--out", str(out), "--only", "C"]) == 0
    assert "没有来源记录" in capsys.readouterr().err
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["beats"]["B"] == {"model": None, "voice": None, "speed": None, "policy_ok": False, "unknown": True}
    assert manifest["beats"]["C"]["voice"] == "Kore" and manifest["beats"]["C"].get("unknown") is None
    assert manifest["policy_ok"] is False
    assert (manifest["model"], manifest["voice"], manifest["speed"]) == (None, None, None)


def test_beat_with_audio_missing_from_manifest_is_unknown(tmp_path, clean_env, capsys):
    clean_env.setattr(synth, "default_speak", fake_speak())
    scene = tmp_path / "two.scene.yaml"
    scene.write_text(TWO_BEATS, encoding="utf-8")
    out = tmp_path / "audio"
    assert main(["synth", str(scene), "--out", str(out)]) == 0  # 两个 beat 都是 Kore
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    del manifest["beats"]["B"]  # manifest 里漏了 B，但 B.wav 还在
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # 有发布说明：按说明（sha256 对得上 B.wav）把 B 的来源恢复成 Kore
    assert main(["synth", str(scene), "--out", str(out), "--only", "C"]) == 0
    restored = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert restored["beats"]["B"]["voice"] == "Kore" and restored["policy_ok"] is True
    capsys.readouterr()

    # 没有说明（加缓存之前的目录）：来源不可知
    del restored["beats"]["B"]
    (out / "manifest.json").write_text(json.dumps(restored), encoding="utf-8")
    shutil.rmtree(out / ".cache" / "beats")

    assert main(["synth", str(scene), "--out", str(out), "--only", "C"]) == 0
    assert "没有来源记录" in capsys.readouterr().err
    after = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert after["beats"]["B"]["unknown"] is True and after["policy_ok"] is False
