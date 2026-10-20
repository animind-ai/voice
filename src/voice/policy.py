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
# 每次运行开头（heal）把每段有音频的 beat 的 wav、timing.json、两份说明合起来核对（check_beat）：
# - repair：找 sha256 和现在的 wav 对得上的那份说明（cues 和 timing.json 一致的优先，其次 beats/、beats-next/）。
#   timing.json 的 cues / total 和它不一样（换了 wav、还没写 timing.json 就断了）就按说明原子地重写 timing.json，
#   清掉对齐结果；说明还在 beats-next 就提交到 beats/。
# - reconcile：按这份说明纠正 manifest 的记录；记录是 pending / unknown、或者没有能对上 wav 的说明，就记成来源不明。
# 没有说明能对上 wav（手动换过 wav、说明坏了）就修不好：manifest 记成来源不明，时间报告不收它（timing_report），
# align 拒绝对齐，synth 报错并以非零退出，直到这一段被重合成。
# 所以在任何一步被打断，manifest 都不会把不合规的音频说成合规，时间报告也不会发布和 wav 对不上的 cues；
# 下一次运行（哪怕只合成别的 beat）会把能修的修好。没有任何说明的旧目录（加缓存之前合成的）保留原来的记录。


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


@dataclass(frozen=True)
class BeatCheck:
    """一段的 wav、timing.json 和两份说明（beats/、beats-next/）合起来看是不是一致的一套（check_beat）。"""
    problem: str = ""  # 非空：修不好（wav 和哪份说明都对不上、没有 wav……），只能重合成这一段
    stamp: dict | None = None  # sha256 和 wav 对得上的说明；None 且没有 problem：没有说明的旧目录，无从核对
    from_next: bool = False  # 对得上的说明还在 beats-next（发布被打断，没提交）
    timing_ok: bool = True  # timing.json 的 cues / total 和这份说明一致
    stale_next: bool = False  # beats-next 里还有一份对不上 wav 的预写说明（换 wav 之前被打断）

    @property
    def publishable(self) -> bool:
        """timing.json 的 cues 就是这个 wav 的：能写进时间报告。"""
        return not self.problem and self.timing_ok

    @property
    def clean(self) -> bool:
        """什么都不用修：能对齐。"""
        return self.publishable and not self.from_next and not self.stale_next

    def describe(self) -> str:
        if self.problem:
            return self.problem
        if not self.timing_ok:
            return "timing.json 的 cues / total 和 wav 对不上（上次发布被打断）"
        return "发布说明还没提交（上次发布被打断）"


def _timing(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _usable(stamp: dict | None, digest: str) -> bool:
    """说明记的就是这个 wav，而且带着恢复 timing.json 和来源要用的字段。"""
    return (stamp is not None and stamp.get("wav_sha256") == digest and all(k in stamp for k in PARTS)
            and isinstance(stamp.get("cues"), list) and isinstance(stamp.get("total"), (int, float)))


def check_beat(out_dir: Path, beat_id: str) -> BeatCheck:
    """wav、timing.json、说明三样一起核对（见上面的说明）。"""
    wav, timing_path = out_dir / f"{beat_id}.wav", out_dir / f"{beat_id}.timing.json"
    committed = _stamp(beats_dir(out_dir) / f"{beat_id}.json")
    pending = _stamp(next_dir(out_dir) / f"{beat_id}.json")
    if not wav.exists():
        return BeatCheck(problem=f"有 {timing_path.name}、没有 {wav.name}" if timing_path.exists()
                         else f"还没有合成（没有 {wav.name} 和 {timing_path.name}）")
    if committed is None and pending is None:
        return BeatCheck()  # 加缓存之前合成的：没有说明，保持原样
    digest = sha256_file(wav)
    candidates = [(s, is_next) for s, is_next in ((committed, False), (pending, True)) if _usable(s, digest)]
    if not candidates:
        return BeatCheck(problem=f"{wav.name} 和发布说明都对不上（被手动换过，或者说明坏了），不知道它的 cues 和来源")
    timing = _timing(timing_path)

    def in_sync(stamp: dict) -> bool:
        return timing is not None and timing.get("cues") == stamp["cues"] and timing.get("total") == stamp["total"]

    stamp, is_next = next((c for c in candidates if in_sync(c[0])), candidates[0])
    return BeatCheck(stamp=stamp, from_next=is_next, timing_ok=in_sync(stamp), stale_next=pending is not None and not is_next)


def repair(out_dir: Path) -> tuple[list[str], dict[str, str]]:
    """按和 wav 对得上的说明把 timing.json 和说明修好，返回 (修了什么, {修不好的 beat: 原因})。

    timing.json 和说明不一致：按说明原子地重写成只有 cues / total（对齐结果是旧音频的，清掉）；
    对得上的说明还在 beats-next：提交到 beats/；beats-next 里对不上 wav 的预写说明删掉。manifest 由 reconcile 修。
    """
    repaired, broken = [], {}
    if not out_dir.is_dir():
        return repaired, broken
    for beat_id in sorted(_present(out_dir)):
        check = check_beat(out_dir, beat_id)
        if check.problem:
            broken[beat_id] = check.problem
            continue
        if check.stamp is None or check.clean:
            continue
        notes = []
        if not check.timing_ok:
            timing = {"cues": check.stamp["cues"], "total": check.stamp["total"]}
            write_text_atomic(out_dir / f"{beat_id}.timing.json", json.dumps(timing, ensure_ascii=False, indent=1))
            notes.append(f"timing.json 按发布说明恢复成 {len(timing['cues'])} 句、{timing['total']:.1f}s，"
                         "清掉过期的对齐结果（要重新 align）")
        if check.from_next:
            write_text_atomic(beats_dir(out_dir) / f"{beat_id}.json", json.dumps(check.stamp))
            notes.append("提交了被打断的发布说明")
        (next_dir(out_dir) / f"{beat_id}.json").unlink(missing_ok=True)
        if notes:
            repaired.append(f"{beat_id}: {'；'.join(notes)}")
    return repaired, broken


def broken_beats(out_dir: Path) -> dict[str, str]:
    """目录里有音频、但 timing.json 不能当作这个 wav 的 cues 的 beat → 原因。"""
    if not out_dir.is_dir():
        return {}
    checks = {beat_id: check_beat(out_dir, beat_id) for beat_id in sorted(_present(out_dir))}
    return {beat_id: check.describe() for beat_id, check in checks.items() if not check.publishable}


def reconcile(out_dir: Path) -> list[str]:
    """按说明纠正 manifest（见上面的说明）。有改动才写，返回改了哪些 beat 的说明文字。"""
    if not out_dir.is_dir():
        return []
    old = _read(out_dir)
    records = _beat_records(out_dir)
    changed = []
    for beat_id in sorted(_present(out_dir)):
        record = records.get(beat_id)
        check = check_beat(out_dir, beat_id)
        if check.stamp is not None:
            fixed = _record(check.stamp["model"], check.stamp["voice"], check.stamp["speed"])
        elif check.problem or record is None or record.get("unknown") or record.get("pending"):
            fixed = dict(UNKNOWN)
        else:
            continue  # 没有说明的旧目录：保留原来的记录
        if record != fixed:
            records[beat_id] = fixed
            changed.append(f"{beat_id}: {(record or {}).get('voice')} → {fixed['voice'] or '来源不明'}")
    if changed:
        _store(out_dir, records, old.get("lang"), old.get("sources"))
    return changed


def heal(out_dir: Path) -> tuple[list[str], list[str], dict[str, str]]:
    """每次 synth / pitch --fix 开头：先 repair，再 reconcile。返回 (修了的文件, 修了的 manifest 记录, 修不好的 beat)。"""
    repaired, broken = repair(out_dir)
    return repaired, reconcile(out_dir), broken
