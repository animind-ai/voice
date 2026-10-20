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

from .ttscache import beats_dir, next_dir, sha256_file, write_text_atomic

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
        return complies(self.model, self.voice, self.speed)

    def policy_problems(self) -> list[str]:
        out = []
        if self.model != REQUIRED_MODEL:
            out.append(f"model={self.model}（应为 {REQUIRED_MODEL}）")
        if self.voice != REQUIRED_VOICE:
            out.append(f"voice={self.voice}（应为 {REQUIRED_VOICE}）")
        if abs(self.speed - REQUIRED_SPEED) > 1e-6:
            out.append(f"speed={self.speed:g}（应为 {REQUIRED_SPEED:g}）")
        return out


def complies(model: str | None, voice: str | None, speed: float | None) -> bool:
    """这组 model / voice / speed 合不合发布政策。"""
    return model == REQUIRED_MODEL and voice == REQUIRED_VOICE and speed is not None and abs(speed - REQUIRED_SPEED) <= 1e-6


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
UNKNOWN = {"model": None, "voice": None, "speed": None, "policy_ok": False, "unknown": True}

# manifest 和音频怎么保持一致（synth / pitch --fix 都在目录锁里做）：
#
# 1. 换掉一段的 wav 之前，先原子地把 manifest 里这一段记成 pending（来源不明、不合规）——预写；
# 2. 再写 .cache/beats-next/<beat>.json（这次要发布的 wav 的 sha256 + model / voice / speed），然后换 wav、写 timing.json、
#    写 .cache/beats/<beat>.json（同样带 sha256 和来源）；
# 3. 最后把实际的 model / voice / speed 提交进 manifest。
#
# 每次运行开头 reconcile：目录里每段音频，用 sha256 和现在的 wav 对得上的那份说明（beats/ 优先，其次 beats-next/）
# 纠正 manifest 的记录；记录是 pending / unknown、或者说明和 wav 对不上，又没有能对上的说明，就记成来源不明。
# 所以在任何一步被打断，manifest 都不会把不合规的音频说成合规；下一次运行（哪怕只合成别的 beat）会把记录修好。
# 没有任何说明的旧目录（加缓存之前合成的）保留原来的记录。


def _read(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}  # 写到一半的 manifest 不会出现（原子替换）；真坏了就当没有，下面按来源不明重建
    return data if isinstance(data, dict) else {}


def _beat_records(out_dir: Path) -> dict[str, dict]:
    """manifest 里每个 beat 实际用的 model / voice / speed。旧格式（只有整目录一组值）按已有的 timing.json 摊到每个 beat。"""
    old = _read(out_dir)
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


def _present(out_dir: Path) -> set[str]:
    """目录里有音频（timing.json 或 wav）的 beat；跳过临时文件（以 . 开头）。"""
    names = {p.name[: -len(".timing.json")] for p in out_dir.glob("*.timing.json")}
    names |= {p.stem for p in out_dir.glob("*.wav")}
    return {n for n in names if not n.startswith(".")}


def _store(out_dir: Path, records: dict[str, dict], lang: str | None, sources: dict | None) -> list[str]:
    """汇总目录级字段，原子写出 manifest，返回提醒。有音频却没有记录的 beat 记成来源不明。"""
    # 目录里有音频、却没有来源记录的 beat（升级前生成、或者 manifest 丢了）：声音不可知，不能声明合规
    for beat_id in sorted(_present(out_dir) - set(records)):
        records[beat_id] = dict(UNKNOWN)
    # 和“混用”一样，只要还有来源不明的 beat 就每次都提醒（synth 每发布一段就写一次 manifest，不能只在第一次发现时说）
    unknown = sorted(beat_id for beat_id, record in records.items() if record.get("unknown"))
    records = dict(sorted(records.items()))
    values = {tuple(r.get(k) for k in PARTS) for r in records.values()}
    mixed = len(values) > 1
    model, voice, speed = next(iter(values)) if values and not mixed else (None, None, None)
    data = {
        "model": model, "voice": voice, "speed": speed, "mixed": mixed,
        "lang": lang, "policy": {"model": REQUIRED_MODEL, "voice": REQUIRED_VOICE, "speed": REQUIRED_SPEED},
        "policy_ok": bool(records) and all(r.get("policy_ok") is True for r in records.values()),
        "sources": sources or {}, "beats": records,
    }
    write_text_atomic(out_dir / MANIFEST_NAME, json.dumps(data, ensure_ascii=False, indent=1))
    warnings = []
    if unknown:
        warnings.append(f"这些 beat 有音频、但没有来源记录（升级前生成、manifest 丢了，或者上次发布被打断），声音不可知，不能算合规：{'、'.join(unknown)}。"
                        "请把这些 beat 重合成")
    if mixed:
        described = "；".join(f"{m} / {v} / 语速 {s}" for m, v, s in sorted(values, key=str))
        warnings.append(f"目录里的 beat 用了不同的声音（{described}）。只重合成了部分 beat 时，要么全部重合成，要么放到单独的目录")
    return warnings


def _record(model: str | None, voice: str | None, speed: float | None) -> dict:
    return {"model": model, "voice": voice, "speed": speed, "policy_ok": complies(model, voice, speed)}


def write_manifest(out_dir: Path, choice: VoiceChoice, lang: str, beat_ids: list[str]) -> list[str]:
    """记下这次合成的 beat 实际用的 model / voice / speed，只改动这次合成的那些 beat（提交，见上面的说明）。

    目录级的字段由各 beat 的记录算：所有 beat 的值一样才写 model / voice / speed，否则 mixed 为真；
    policy_ok 要求每个 beat 都合规。返回混用的说明（同一目录里声音不一致），只要还混着就每次都返回。
    """
    records = _beat_records(out_dir)
    for beat_id in beat_ids:
        records[beat_id] = _record(choice.model, choice.voice, choice.speed)
    return _store(out_dir, records, lang, choice.sources)


def mark_pending(out_dir: Path, beat_id: str, lang: str | None = None, sources: dict | None = None) -> None:
    """预写：要换这一段的 wav 了，先把它记成来源不明、不合规。lang / sources 没给就沿用 manifest 里的。"""
    old = _read(out_dir)
    records = _beat_records(out_dir)
    records[beat_id] = {**UNKNOWN, "pending": True}
    _store(out_dir, records, lang or old.get("lang"), sources if sources is not None else old.get("sources"))


def _stamp(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def stamp_provenance(out_dir: Path, beat_id: str) -> tuple[str, dict | None]:
    """这一段现在的 wav 是谁念的：("match", 来源)、("mismatch", None)（有说明但都对不上 wav）或 ("none", None)（没有说明）。"""
    stamps = [s for s in (_stamp(beats_dir(out_dir) / f"{beat_id}.json"), _stamp(next_dir(out_dir) / f"{beat_id}.json")) if s]
    wav = out_dir / f"{beat_id}.wav"
    if not stamps:
        return "none", None
    digest = sha256_file(wav) if wav.exists() else None
    for stamp in stamps:
        if digest is not None and stamp.get("wav_sha256") == digest and all(k in stamp for k in PARTS):
            return "match", {k: stamp[k] for k in PARTS}
    return "mismatch", None


def reconcile(out_dir: Path) -> list[str]:
    """按说明纠正 manifest（见上面的说明）。有改动才写，返回改了哪些 beat 的说明文字。"""
    if not out_dir.is_dir():
        return []
    old = _read(out_dir)
    records = _beat_records(out_dir)
    changed = []
    for beat_id in sorted(_present(out_dir)):
        record = records.get(beat_id)
        status, source = stamp_provenance(out_dir, beat_id)
        if status == "match":
            fixed = _record(source["model"], source["voice"], source["speed"])
        elif status == "mismatch" or record is None or record.get("unknown") or record.get("pending"):
            fixed = dict(UNKNOWN)
        else:
            continue  # 没有说明的旧目录：保留原来的记录
        if record != fixed:
            records[beat_id] = fixed
            changed.append(f"{beat_id}: {(record or {}).get('voice')} → {fixed['voice'] or '来源不明'}")
    if changed:
        _store(out_dir, records, old.get("lang"), old.get("sources"))
    return changed
