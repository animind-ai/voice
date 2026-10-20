"""TTS 声音的取值规则：场景的 style.voice 是准绳。

优先级（高到低）：
1. 命令行 --model / --voice / --speed（显式覆盖，打印一条提示）；
2. 环境变量 GEMINI_TTS_MODEL / GEMINI_TTS_VOICE，只有 --allow-env-voice 或 VOICE_ALLOW_ENV_OVERRIDE=1 时才采用，并打印大声警告；
3. 场景 style.voice 里写的值；
4. 场景没写时的默认值：google/gemini-3.8-flash-tts、Kore、语速 1.0。

有效值和发布政策（scene_spec.quality 里的 REQUIRED_*）不同，就打印 WARNING：这批音频不能发布。
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from scene_spec.quality import REQUIRED_MODEL, REQUIRED_SPEED, REQUIRED_VOICE

ENV_MODEL = "GEMINI_TTS_MODEL"
ENV_VOICE = "GEMINI_TTS_VOICE"
ENV_ALLOW = "VOICE_ALLOW_ENV_OVERRIDE"


@dataclass(frozen=True)
class VoiceChoice:
    model: str
    voice: str
    speed: float
    sources: dict[str, str]  # 每个字段来自哪里：default / scene / env / flag
    notices: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def policy_ok(self) -> bool:
        return self.model == REQUIRED_MODEL and self.voice == REQUIRED_VOICE and abs(self.speed - REQUIRED_SPEED) <= 1e-6

    def policy_problems(self) -> list[str]:
        out = []
        if self.model != REQUIRED_MODEL:
            out.append(f"model={self.model}（应为 {REQUIRED_MODEL}）")
        if self.voice != REQUIRED_VOICE:
            out.append(f"voice={self.voice}（应为 {REQUIRED_VOICE}）")
        if abs(self.speed - REQUIRED_SPEED) > 1e-6:
            out.append(f"speed={self.speed:g}（应为 {REQUIRED_SPEED:g}）")
        return out


def env_override_allowed(environ: Mapping[str, str], flag: bool) -> bool:
    return flag or environ.get(ENV_ALLOW) == "1"


def resolve(doc: dict, *, model: str | None = None, voice: str | None = None, speed: float | None = None,
            allow_env: bool = False, environ: Mapping[str, str] | None = None) -> VoiceChoice:
    environ = os.environ if environ is None else environ
    scene = (doc.get("style") or {}).get("voice") or {}
    values = {"model": REQUIRED_MODEL, "voice": REQUIRED_VOICE, "speed": REQUIRED_SPEED}
    sources = {k: "default" for k in values}
    for key in values:
        if key in scene:
            values[key], sources[key] = scene[key], "scene"

    notices, warnings = [], []
    env_allowed = env_override_allowed(environ, allow_env)
    for key, env_name in (("model", ENV_MODEL), ("voice", ENV_VOICE)):
        env_value = environ.get(env_name)
        if not env_value:
            continue
        if env_allowed:
            warnings.append(f"⚠⚠ 环境变量 {env_name}={env_value} 覆盖了场景 style.voice.{key}（{values[key]}）。"
                            f"这是显式允许的覆盖（--allow-env-voice 或 {ENV_ALLOW}=1）。")
            values[key], sources[key] = env_value, "env"
        else:
            notices.append(f"{env_name} 已设置但未采用，场景 style.voice 优先（要用它需 --allow-env-voice 或 {ENV_ALLOW}=1）")

    for key, flag_value in (("model", model), ("voice", voice), ("speed", speed)):
        if flag_value is None:
            continue
        if flag_value != values[key]:
            notices.append(f"命令行 --{key} 覆盖 {values[key]!r}，改用 {flag_value!r}")
        values[key], sources[key] = flag_value, "flag"

    _check_values(values)
    return VoiceChoice(values["model"], values["voice"], float(values["speed"]), sources, notices, warnings)


def _check_values(values: dict) -> None:
    """模型和声音必须是非空字符串，语速必须是正的有限数（NaN 和 Infinity 不接受）。"""
    for key in ("model", "voice"):
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"{key} 应为非空字符串，收到 {values[key]!r}")
    _check_speed(values["speed"], "语速（style.voice.speed 或 --speed）")


def _check_speed(speed: object, where: str) -> None:
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"{where} 应为正的有限数，收到 {speed!r}")


MANIFEST_NAME = "manifest.json"
PARTS = ("model", "voice", "speed")


def _beat_records(out_dir: Path) -> dict[str, dict]:
    """manifest 里每个 beat 实际用的 model / voice / speed。旧格式（只有整目录一组值）按已有的 timing.json 摊到每个 beat。"""
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    old = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(old.get("beats"), dict):
        return dict(old["beats"])
    if "model" not in old:
        return {}
    stems = {p.name[: -len(".timing.json")] for p in out_dir.glob("*.timing.json")}
    return {stem: {**{k: old.get(k) for k in PARTS}, "policy_ok": old.get("policy_ok")} for stem in stems}


def beat_speed(out_dir: Path, beat_id: str, default: float) -> float:
    """重合成一个 beat 用它当初实际的语速（manifest 里的 beats 记录），没有记录就用 default。"""
    record = _beat_records(out_dir).get(beat_id)
    return float(record["speed"]) if record and record.get("speed") is not None else default


def write_manifest(out_dir: Path, choice: VoiceChoice, lang: str, beat_ids: list[str]) -> list[str]:
    """记下这次合成的 beat 实际用的 model / voice / speed，只改动这次合成的那些 beat。

    目录级的字段由各 beat 的记录算：所有 beat 的值一样才写 model / voice / speed，否则 mixed 为真；
    policy_ok 要求每个 beat 都合规。返回混用的说明（同一目录里声音不一致），只要还混着就每次都返回。
    """
    records = _beat_records(out_dir)
    for beat_id in beat_ids:
        records[beat_id] = {"model": choice.model, "voice": choice.voice, "speed": choice.speed,
                            "policy_ok": choice.policy_ok}
    # 目录里有音频、却没有来源记录的 beat（升级前生成、或者 manifest 丢了）：声音不可知，不能声明合规
    present = {p.name[: -len(".timing.json")] for p in out_dir.glob("*.timing.json")}
    present |= {p.stem for p in out_dir.glob("*.wav")}
    unknown = sorted(present - set(records))
    for beat_id in unknown:
        records[beat_id] = {"model": None, "voice": None, "speed": None, "policy_ok": False, "unknown": True}
    records = dict(sorted(records.items()))
    values = {tuple(r.get(k) for k in PARTS) for r in records.values()}
    mixed = len(values) > 1
    model, voice, speed = next(iter(values)) if values and not mixed else (None, None, None)
    data = {
        "model": model, "voice": voice, "speed": speed, "mixed": mixed,
        "lang": lang, "policy": {"model": REQUIRED_MODEL, "voice": REQUIRED_VOICE, "speed": REQUIRED_SPEED},
        "policy_ok": bool(records) and all(r.get("policy_ok") for r in records.values()),
        "sources": choice.sources, "beats": records,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    warnings = []
    if unknown:
        warnings.append(f"这些 beat 有音频、但没有来源记录（升级前生成，或 manifest 丢了），声音不可知，不能算合规：{'、'.join(unknown)}。"
                        "请把整集全部重合成")
    if mixed:
        described = "；".join(f"{m} / {v} / 语速 {s}" for m, v, s in sorted(values, key=str))
        warnings.append(f"目录里的 beat 用了不同的声音（{described}）。只重合成了部分 beat 时，要么全部重合成，要么放到单独的目录")
    return warnings
