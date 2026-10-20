"""成片审片：把视频（加场景稿 / 旁白 / 书签时刻）交给 Gemini，写出结构化报告。

报告 JSON 的 `gate` 和 scene-spec 检查器的 `Report.to_dict()` 同形（ok / issues / stats），
方便以后当作流水线的门。模型调用可以换成假的，测试不联网。

费用表跟 synth.py 一样写在这里。价格变了就改 PRICE，并改 README 的「费用」。
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "gemini-4-argon"
# 文字多的数学画面用 high：视频理解文档里高分辨率大约每秒 300 token（含音频），低分辨率大约 100。
VIDEO_TOKENS_PER_SECOND = {"low": 100, "high": 300}
DEFAULT_MAX_CHUNK_SECONDS = 300.0
DEFAULT_OVERLAP_SECONDS = 8.0
MAX_ISSUES_PER_CHUNK = 40
INLINE_MAX_BYTES = 15 * 1024 * 1024

# 标准档，美元 / 百万 token。https://ai.google.dev/gemini-api/docs/pricing （2026-10-20 查阅）。
# until 是涨价生效日，当天起用 *_after。long_at 是长上下文分界（prompt token 超过它用 *_long）。
PRICES: dict[str, dict] = {
    "gemini-4-argon": {
        "until": "2026-12-01",
        "input": 2.0, "output": 10.0,
        "input_after": 4.0, "output_after": 20.0,
    },
    "gemini-3.1-pro-preview": {
        "input": 2.0, "output": 12.0,
        "input_long": 4.0, "output_long": 18.0,
        "long_at": 200_000,
    },
    "gemini-3.8-flash": {
        "until": "2027-01-01",
        "input": 0.75, "output": 3.75,
        "input_after": 1.50, "output_after": 7.50,
    },
}

CATEGORIES = (
    "layout_overlap",
    "out_of_frame",
    "legibility",
    "av_sync",
    "tts_pronunciation",
    "tts_filler",
    "tts_voice_drift",
    "pedagogy",
    "math",
    "seam",
    "other",
)
CATEGORY_CODE = {
    "layout_overlap": "RV100",
    "out_of_frame": "RV110",
    "legibility": "RV120",
    "av_sync": "RV200",
    "tts_pronunciation": "RV210",
    "tts_filler": "RV220",
    "tts_voice_drift": "RV230",
    "pedagogy": "RV300",
    "math": "RV310",
    "seam": "RV400",
    "other": "RV900",
}
SEVERITIES = ("info", "minor", "major", "blocker")
SEVERITY_RANK = {name: i for i, name in enumerate(SEVERITIES)}
DEFAULT_THRESHOLDS = {
    "fail_severities": ["blocker", "major"],
    "max_counts": {},
    "fail_categories": {},
}
SECRET_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY")


class ReviewError(Exception):
    """审片没能完成（凭证、视频、模型一直返回坏 JSON）。"""

    def __init__(self, message: str, calls: list | None = None, token_source: str = "usage"):
        super().__init__(message)
        self.calls = calls or []
        self.token_source = token_source


class ReviewSchemaError(ReviewError):
    """模型返回的 JSON 不符合 v1 schema。"""


@dataclass(frozen=True)
class Chunk:
    index: int
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Turn:
    """一次审片请求。calls 里每一对 (input, output) 是一次实际的模型 HTTP 调用（含重试）。"""

    text: str
    calls: list[tuple[int, int]] = field(default_factory=lambda: [(0, 0)])
    token_source: str = "usage"

    @property
    def attempts(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(pair[0] for pair in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(pair[1] for pair in self.calls)


def redact(message: str) -> str:
    for name in SECRET_ENVS:
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "***")
    return message


def _redact_obj(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_obj(item) for key, item in value.items()}
    return value


def normalize_model(model: str) -> str:
    """OpenRouter 的名字带 google/ 前缀。审片走 google.genai，用去掉前缀的模型 id。"""
    if model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def prompt_dir(version: str = PROMPT_VERSION) -> Path:
    return Path(__file__).resolve().parent / "prompts" / "review" / version


def load_prompt(version: str = PROMPT_VERSION) -> tuple[str, str, dict]:
    root = prompt_dir(version)
    if not root.is_dir():
        raise ReviewError(f"没有这个提示词版本：{version}")
    system = (root / "system.md").read_text(encoding="utf-8")
    user = (root / "user.md").read_text(encoding="utf-8")
    schema = json.loads((root / "response.schema.json").read_text(encoding="utf-8"))
    return system, user, schema


def plan_chunks(duration: float, max_chunk: float, overlap: float) -> list[Chunk]:
    """把整集切成重叠的窗口。短于 max_chunk 的片子只有一段。"""
    if duration <= 0:
        raise ReviewError("视频时长必须大于 0")
    if max_chunk <= 0:
        raise ReviewError("分段长度必须大于 0")
    if overlap < 0 or overlap >= max_chunk:
        raise ReviewError("重叠必须大于等于 0，并且小于分段长度")
    if duration <= max_chunk:
        return [Chunk(0, 0.0, round(duration, 3))]
    step = max_chunk - overlap
    chunks: list[Chunk] = []
    start = 0.0
    while True:
        end = min(duration, start + max_chunk)
        chunks.append(Chunk(len(chunks), round(start, 3), round(end, 3)))
        if end >= duration - 1e-3:
            break
        start += step
    return chunks


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        raise ReviewError(f"读不到视频时长（需要 ffprobe）：{detail}") from exc
    text = out.stdout.strip()
    try:
        duration = float(text)
    except ValueError as exc:
        raise ReviewError(f"ffprobe 没有返回时长：{text!r}") from exc
    if duration <= 0:
        raise ReviewError(f"视频时长是 {duration}")
    return duration


def cut_clip(src: Path, start: float, end: float, dest: Path) -> None:
    """切出 [start, end)。重编码，-ss 放在 -i 前面时 ffmpeg 仍会丢弃关键帧之前的多余画面。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.05, end - start)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(dest)],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        raise ReviewError(f"切分视频失败：{detail}") from exc


def price_call(model: str, input_tokens: int, output_tokens: int, today: date | None = None) -> tuple[float, bool]:
    """一次调用的估算美元，以及价目表里有没有这个模型。不认识的模型按默认模型计价，known 为 False。"""
    today = today or date.today()
    spec = PRICES.get(model)
    known = spec is not None
    if spec is None:
        spec = PRICES[DEFAULT_MODEL]
    rate_in, rate_out = spec["input"], spec["output"]
    until = spec.get("until")
    if until and today >= date.fromisoformat(until):
        rate_in, rate_out = spec["input_after"], spec["output_after"]
    long_at = spec.get("long_at")
    if long_at and input_tokens > long_at:
        rate_in, rate_out = spec["input_long"], spec["output_long"]
    usd = (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
    return usd, known


def estimate_tokens(text: str, seconds: float, resolution: str) -> tuple[int, int]:
    """API 没返回 usage 时的粗估。输出按一份短 JSON 报告估，宁多不少。"""
    per_second = VIDEO_TOKENS_PER_SECOND[resolution]
    return int(len(text) * 1.5) + int(round(seconds * per_second)), 800


def load_thresholds(path: Path | None) -> dict:
    thresholds = {
        "fail_severities": list(DEFAULT_THRESHOLDS["fail_severities"]),
        "max_counts": dict(DEFAULT_THRESHOLDS["max_counts"]),
        "fail_categories": dict(DEFAULT_THRESHOLDS["fail_categories"]),
    }
    if path is None:
        return thresholds
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"读不到阈值文件：{exc}") from exc
    if not isinstance(data, dict):
        raise ReviewError("阈值文件的顶层必须是 JSON 对象")
    unknown = set(data) - set(thresholds)
    if unknown:
        raise ReviewError(f"阈值文件有不认识的字段：{sorted(unknown)}")
    if "fail_severities" in data:
        bad = [item for item in data["fail_severities"] if item not in SEVERITY_RANK]
        if not isinstance(data["fail_severities"], list) or bad:
            raise ReviewError(f"fail_severities 只能是 {list(SEVERITIES)}")
        thresholds["fail_severities"] = list(data["fail_severities"])
    if "max_counts" in data:
        counts = data["max_counts"]
        if not isinstance(counts, dict):
            raise ReviewError("max_counts 必须是对象")
        for key, value in counts.items():
            if key not in SEVERITY_RANK or isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReviewError(f"max_counts.{key} 必须是非负整数，级别只能是 {list(SEVERITIES)}")
        thresholds["max_counts"] = dict(counts)
    if "fail_categories" in data:
        cats = data["fail_categories"]
        if not isinstance(cats, dict):
            raise ReviewError("fail_categories 必须是对象")
        for key, value in cats.items():
            if key not in CATEGORY_CODE or value not in SEVERITY_RANK:
                raise ReviewError(f"fail_categories.{key}={value!r} 不合法")
        thresholds["fail_categories"] = dict(cats)
    return thresholds


def _as_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReviewSchemaError(f"{name} 必须是数字")
    if value < 0:
        raise ReviewSchemaError(f"{name} 不能为负")
    return float(value)


def _as_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReviewSchemaError(f"{name} 必须是非空字符串")
    return value.strip()[:800]


def extract_json(text: str) -> dict:
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise ReviewSchemaError("输出不是 JSON 对象") from None
        try:
            data = json.loads(body[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ReviewSchemaError(f"输出不是 JSON：{exc}") from exc
    if not isinstance(data, dict):
        raise ReviewSchemaError("输出的顶层必须是 JSON 对象")
    return data


def parse_model_output(text: str, *, chunk_len: float) -> dict:
    data = extract_json(text)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ReviewSchemaError("summary 必须是非空字符串")
    raw_issues = data.get("issues")
    if not isinstance(raw_issues, list):
        raise ReviewSchemaError("issues 必须是数组")
    if len(raw_issues) > MAX_ISSUES_PER_CHUNK:
        raise ReviewSchemaError(f"issues 有 {len(raw_issues)} 条，最多 {MAX_ISSUES_PER_CHUNK} 条，只留会让观众卡住的")
    issues = []
    for index, raw in enumerate(raw_issues):
        if not isinstance(raw, dict):
            raise ReviewSchemaError(f"issues[{index}] 必须是对象")
        category = raw.get("category")
        severity = raw.get("severity")
        if category not in CATEGORY_CODE:
            raise ReviewSchemaError(f"issues[{index}].category={category!r} 不在 {list(CATEGORIES)}")
        if severity not in SEVERITY_RANK:
            raise ReviewSchemaError(f"issues[{index}].severity={severity!r} 不在 {list(SEVERITIES)}")
        start = _as_float(raw.get("start"), f"issues[{index}].start")
        end = _as_float(raw.get("end"), f"issues[{index}].end")
        if end < start:
            raise ReviewSchemaError(f"issues[{index}] 的 end 早于 start")
        if start > chunk_len + 0.05:
            raise ReviewSchemaError(
                f"issues[{index}].start={start:.3f} 超出本段 {chunk_len:.3f}s，时间要从本段开头算")
        end = min(end, chunk_len)
        issue = {
            "category": category,
            "severity": severity,
            "start": round(start, 3),
            "end": round(end, 3),
            "evidence": _as_text(raw.get("evidence"), f"issues[{index}].evidence"),
            "suggested_fix": _as_text(raw.get("suggested_fix"), f"issues[{index}].suggested_fix"),
        }
        for key in ("beat", "sentence"):
            if key in raw and raw[key] not in (None, ""):
                if not isinstance(raw[key], str):
                    raise ReviewSchemaError(f"issues[{index}].{key} 必须是字符串")
                issue[key] = raw[key].strip()[:80]
        issues.append(issue)
    return {"summary": summary.strip()[:600], "issues": issues}


def _overlap_fraction(a: dict, b: dict) -> float:
    overlap = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if overlap <= 0:
        return 0.0
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    if shorter <= 1e-6:
        return 1.0
    return overlap / shorter


def dedupe_issues(issues: list[dict]) -> list[dict]:
    """重叠窗口里同一类、时间重叠过半的问题合并成一条，留下更严重的那条证据。"""
    ordered = sorted(issues, key=lambda item: (item["start"], -SEVERITY_RANK[item["severity"]]))
    kept: list[dict] = []
    for issue in ordered:
        match = next((prev for prev in kept
                      if prev["category"] == issue["category"] and _overlap_fraction(prev, issue) >= 0.5), None)
        if match is None:
            kept.append(dict(issue))
            continue
        if SEVERITY_RANK[issue["severity"]] > SEVERITY_RANK[match["severity"]]:
            match["severity"] = issue["severity"]
            match["evidence"] = issue["evidence"]
            match["suggested_fix"] = issue["suggested_fix"]
            match["chunk"] = issue["chunk"]
            for key in ("beat", "sentence"):
                if issue.get(key):
                    match[key] = issue[key]
        match["start"] = round(min(match["start"], issue["start"]), 3)
        match["end"] = round(max(match["end"], issue["end"]), 3)
    return kept


def issue_fails(issue: dict, thresholds: dict) -> bool:
    if issue["severity"] in thresholds["fail_severities"]:
        return True
    floor = thresholds["fail_categories"].get(issue["category"])
    return floor is not None and SEVERITY_RANK[issue["severity"]] >= SEVERITY_RANK[floor]


def decide(issues: list[dict], thresholds: dict) -> tuple[bool, list[str]]:
    """返回 (passed, 额外的门槛说明)。额外说明是条数超限，不是某一条问题本身。"""
    counts: dict[str, int] = {}
    for issue in issues:
        counts[issue["severity"]] = counts.get(issue["severity"], 0) + 1
    violations = []
    for severity, limit in thresholds["max_counts"].items():
        if counts.get(severity, 0) > limit:
            violations.append(f"{severity} 有 {counts[severity]} 条，超过上限 {limit}")
    failed = any(issue_fails(issue, thresholds) for issue in issues) or bool(violations)
    return not failed, violations


def project_gate(issues: list[dict], thresholds: dict, violations: list[str]) -> dict:
    gate_issues = []
    stats: dict[str, int] = {}
    for issue in issues:
        stats[issue["category"]] = stats.get(issue["category"], 0) + 1
        stats[issue["severity"]] = stats.get(issue["severity"], 0) + 1
        failing = issue_fails(issue, thresholds)
        where = []
        if issue.get("beat"):
            where.append(str(issue["beat"]))
        if issue.get("sentence"):
            where.append(str(issue["sentence"]))
        prefix = (" ".join(where) + " ") if where else ""
        gate_issues.append({
            "level": "error" if failing else "warning",
            "code": issue["code"],
            "path": f"t={issue['start']:.3f}-{issue['end']:.3f}",
            "message": f"{prefix}{issue['category']}: {issue['evidence']} 建议：{issue['suggested_fix']}",
        })
    for violation in violations:
        gate_issues.append({"level": "error", "code": "RV001", "path": "", "message": violation})
        stats["threshold"] = stats.get("threshold", 0) + 1
    ok = not any(item["level"] == "error" for item in gate_issues)
    return {"ok": ok, "issues": gate_issues, "stats": stats}


def render_user(template: str, **values: str) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _fmt_clock(seconds: float) -> str:
    whole = int(seconds)
    frac = int(round((seconds - whole) * 10))
    if frac == 10:
        whole += 1
        frac = 0
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}.{frac}"
    return f"{minutes:02d}:{secs:02d}.{frac}"


def render_markdown(report: dict) -> str:
    title = report.get("title") or Path(report["episode"]).stem
    status = "通过" if report["passed"] else "未通过"
    counts = report["counts"]
    bits = [f"{counts.get(name, 0)} {name}" for name in reversed(SEVERITIES) if counts.get(name)]
    count_text = "、".join(bits) if bits else "没有问题"
    lines = [
        f"# 审片：{title}",
        "",
        f"**{status}**（{report['model']}，提示词 {report['prompt_version']}，时长 {_fmt_clock(report['duration_s'])}）",
        "",
        report["summary"],
        "",
        f"问题：{count_text}。",
        "",
    ]
    if report["issues"]:
        lines.append("## 问题")
        lines.append("")
        for issue in report["issues"]:
            where = f"{_fmt_clock(issue['start'])}–{_fmt_clock(issue['end'])}"
            lines.append(f"- **{issue['severity']}** `{issue['category']}` `{issue['code']}` {where}")
            lines.append(f"  {issue['evidence']}")
            lines.append(f"  建议：{issue['suggested_fix']}")
        lines.append("")
    usage = report["usage"]
    lines.append(f"调用 {usage['calls']} 次，约 ${usage['est_usd']:.4f}。明细：`{usage['log']}`。")
    lines.append("")
    return "\n".join(lines)


def _check_line(check: dict) -> str:
    missing = check.get("missing") or []
    extra = check.get("extra") or []
    ratio = check.get("ratio")
    if not missing and not extra and (ratio is None or ratio >= 0.85):
        return ""
    heard = str(check.get("heard") or "")
    if len(heard) > 180:
        heard = heard[:180] + "…"
    return f"（对齐：匹配 {ratio}；没听到 {missing}；多出 {extra}；听到 {heard}）"


def build_context(spec: Path | None, audio: Path | None, lang: str) -> tuple[str, str | None, str | None]:
    """旁白、书签和回听提示。返回 (文本, episode_id, 标题)。时刻是 beat 内部的，不是整集时间轴。"""
    if spec is None:
        return "（没有场景文件。只根据这一段的画面和声音判断，不要编造稿子。）", None, None
    from scene_spec import localize
    from scene_spec.loader import load_file

    from . import script

    doc = localize(load_file(spec), lang)
    meta = doc.get("meta") or {}
    pedagogy = doc.get("pedagogy") or {}
    lines = [f"标题：{meta.get('title') or spec.stem}", f"集 id：{meta.get('id') or ''}"]
    if pedagogy.get("insight"):
        lines.append(f"这一集要讲清楚：{pedagogy['insight']}")
    if pedagogy.get("misconception"):
        lines.append(f"学生容易错在：{pedagogy['misconception']}")
    lines.append("下面每句的秒数是这一 beat 内部的时刻，不是整集时间轴。整集时间以视频为准。")
    for beat in script.beats(doc):
        timing = None
        if audio is not None:
            timing_path = audio / f"{beat.id}.timing.json"
            if timing_path.exists():
                timing = json.loads(timing_path.read_text(encoding="utf-8"))
        cues = (timing or {}).get("cues") or []
        marks = (timing or {}).get("marks") or {}
        checks = (timing or {}).get("check") or {}
        lines.append(f"\n## beat {beat.id}")
        for index, sentence in enumerate(beat.sentences):
            cue = cues[index] if index < len(cues) else {}
            span = f" [{cue.get('start')}–{cue.get('end')}s]" if cue else ""
            mark_text = ""
            if sentence.key in marks:
                mark_text = " 书签 " + "，".join(f"{name}@{moment}" for name, moment in marks[sentence.key].items())
            hint = _check_line(checks.get(sentence.key) or {})
            lines.append(f"- {sentence.key}{span} ({sentence.pace}) {sentence.spoken}{mark_text}{hint}")
    raw = spec.read_text(encoding="utf-8")
    if len(raw) > 20_000:
        raw = raw[:20_000] + "\n…（场景文件已截断）"
    lines.append("\n## 场景文件\n")
    lines.append(raw)
    return "\n".join(lines), meta.get("id"), meta.get("title")


def _window_text(chunk: Chunk) -> str:
    return f"{_fmt_clock(chunk.start)}–{_fmt_clock(chunk.end)}"


def total_usage(path: Path) -> tuple[int, float]:
    """用量记录里的模型调用次数（每行一次 HTTP）和估算总费用。"""
    if not path.exists():
        return 0, 0.0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows), sum(row.get("est_usd", 0.0) for row in rows)


def _append_usage(path: Path, episode: str, chunk: Chunk, model: str, turn: Turn, prompt_version: str,
                  today: date) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    stamp = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        for attempt, (input_tokens, output_tokens) in enumerate(turn.calls, start=1):
            usd, known = price_call(model, input_tokens, output_tokens, today)
            record = {
                "time": stamp, "episode": episode, "chunk": chunk.index, "attempt": attempt,
                "model": model, "prompt_version": prompt_version, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "est_usd": round(usd, 6), "token_source": turn.token_source,
                "price_known": known,
            }
            handle.write(json.dumps(_redact_obj(record), ensure_ascii=False) + "\n")
            spent += usd
    return spent


def _clip_paths(video: Path, chunks: list[Chunk], cut, tmp: Path) -> list[Path]:
    if len(chunks) == 1 and chunks[0].start == 0:
        return [video]
    paths = []
    for chunk in chunks:
        dest = tmp / f"chunk-{chunk.index:02d}.mp4"
        cut(video, chunk.start, chunk.end, dest)
        paths.append(dest)
    return paths


def review_episode(
    video: Path,
    *,
    reviewer,
    out_json: Path,
    model: str,
    spec: Path | None = None,
    audio: Path | None = None,
    lang: str = "zh",
    thresholds: dict | None = None,
    max_chunk_seconds: float = DEFAULT_MAX_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    prompt_version: str = PROMPT_VERSION,
    probe=None,
    cut=None,
    schema_attempts: int = 3,
    today: date | None = None,
    log=print,
) -> dict:
    """审完一整集，写出 JSON 和同名 Markdown。reviewer(video, system, user, chunk) -> Turn。"""
    if schema_attempts < 1:
        raise ReviewError("schema 重试次数至少为 1")
    probe = probe or probe_duration
    cut = cut or cut_clip
    video = video.resolve()
    if not video.exists():
        raise ReviewError(f"找不到视频：{video}")
    today = today or date.today()
    model = normalize_model(model)
    thresholds = thresholds or load_thresholds(None)
    system, user_template, schema = load_prompt(prompt_version)
    context, episode_id, title = build_context(spec, audio, lang)
    duration = probe(video)
    chunks = plan_chunks(duration, max_chunk_seconds, overlap_seconds)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2)
    log_path = out_json.parent / "review_usage.jsonl"
    summaries: list[str] = []
    found: list[dict] = []
    chunk_meta: list[dict] = []

    import tempfile

    with tempfile.TemporaryDirectory(prefix="animind-review-") as tmp_name:
        paths = _clip_paths(video, chunks, cut, Path(tmp_name))
        for chunk, clip in zip(chunks, paths):
            base_user = render_user(
                user_template, CONTEXT=context, INDEX=str(chunk.index + 1), COUNT=str(len(chunks)),
                WINDOW=_window_text(chunk), DURATION=_fmt_clock(duration), SCHEMA=schema_text,
            )
            user = base_user
            parsed = None
            chunk_calls = 0
            last_error: ReviewSchemaError | None = None
            for _ in range(schema_attempts):
                try:
                    turn = reviewer(video=clip, system=system, user=user, chunk=chunk)
                except ReviewError as exc:
                    if exc.calls:
                        _append_usage(log_path, video.name, chunk, model,
                                      Turn(text="", calls=list(exc.calls), token_source=exc.token_source),
                                      prompt_version, today)
                    raise
                if not isinstance(turn, Turn) or not turn.calls:
                    raise ReviewError("reviewer 必须返回带用量的 Turn")
                chunk_calls += turn.attempts
                _append_usage(log_path, video.name, chunk, model, turn, prompt_version, today)
                try:
                    parsed = parse_model_output(turn.text, chunk_len=chunk.length)
                    break
                except ReviewSchemaError as exc:
                    last_error = exc
                    user = base_user + f"\n\n上一次的输出不能用：{exc}\n请只返回符合 schema 的 JSON。"
            if parsed is None:
                raise ReviewSchemaError(f"第 {chunk.index + 1} 段试了 {schema_attempts} 次仍不符合格式：{last_error}")
            summaries.append(parsed["summary"])
            for issue in parsed["issues"]:
                issue["start"] = round(min(duration, issue["start"] + chunk.start), 3)
                issue["end"] = round(min(duration, issue["end"] + chunk.start), 3)
                issue["chunk"] = chunk.index
                found.append(issue)
            chunk_meta.append({"index": chunk.index, "start": chunk.start, "end": chunk.end, "calls": chunk_calls})
            log(f"  段 {chunk.index + 1}/{len(chunks)} {_window_text(chunk)}：{len(parsed['issues'])} 条")

    merged = dedupe_issues(found)
    for number, issue in enumerate(merged, start=1):
        issue["id"] = f"i{number}"
        issue["code"] = CATEGORY_CODE[issue["category"]]
    passed, violations = decide(merged, thresholds)
    gate = project_gate(merged, thresholds, violations)
    counts = {name: sum(1 for issue in merged if issue["severity"] == name) for name in SEVERITIES}
    new_lines = sum(item["calls"] for item in chunk_meta)
    calls, est_usd = _sum_new_lines(log_path, new_lines)
    input_tokens, output_tokens = _sum_tokens(log_path, new_lines)
    if summaries:
        summary = summaries[0] if len(summaries) == 1 else "\n".join(dict.fromkeys(summaries))
    else:
        summary = "没有可看的片段。"
    report = {
        "schema": "animind.review/v1",
        "prompt_version": prompt_version,
        "episode": video.name,
        "episode_id": episode_id,
        "title": title,
        "model": model,
        "duration_s": round(duration, 3),
        "passed": passed,
        "ok": gate["ok"],
        "summary": summary[:600],
        "thresholds": thresholds,
        "counts": counts,
        "issues": merged,
        "gate": gate,
        "usage": {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_usd": round(est_usd, 6),
            "log": log_path.name,
        },
        "chunks": chunk_meta,
        "sources": {
            "video": str(video),
            "spec": str(spec) if spec else None,
            "audio": str(audio) if audio else None,
            "prompt_version": prompt_version,
        },
    }
    report = _redact_obj(report)
    validate_report(report)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_json.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def validate_report(report: dict) -> None:
    """写出前核对报告形状，保证 scene-spec 以后能直接读 gate。"""
    required = ("schema", "prompt_version", "episode", "model", "duration_s", "passed", "ok", "summary",
                "thresholds", "counts", "issues", "gate", "usage", "chunks")
    missing = [key for key in required if key not in report]
    if missing or report.get("schema") != "animind.review/v1":
        raise ReviewError(f"报告缺字段：{missing}")
    gate = report["gate"]
    if report["passed"] != report["ok"] or gate.get("ok") != report["ok"]:
        raise ReviewError("passed 和 gate.ok 不一致")
    if set(gate) != {"ok", "issues", "stats"}:
        raise ReviewError("gate 必须只有 ok、issues、stats")
    for issue in report["issues"]:
        for key in ("id", "category", "code", "severity", "start", "end", "evidence", "suggested_fix", "chunk"):
            if key not in issue:
                raise ReviewError(f"问题缺 {key}")
        if issue["category"] not in CATEGORY_CODE or issue["code"] != CATEGORY_CODE[issue["category"]]:
            raise ReviewError("code 和 category 不一致")
        if issue["severity"] not in SEVERITY_RANK or issue["end"] < issue["start"]:
            raise ReviewError("报告里的级别或时间不合法")
    for item in gate["issues"]:
        if set(item) != {"level", "code", "path", "message"}:
            raise ReviewError("gate.issues 的字段必须是 level、code、path、message")
        if item["level"] not in {"error", "warning"}:
            raise ReviewError("gate level 不合法")
    if set(report["usage"]) != {"calls", "input_tokens", "output_tokens", "est_usd", "log"}:
        raise ReviewError("usage 字段不完整")


def _sum_new_lines(path: Path, new_lines: int) -> tuple[int, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fresh = rows[-new_lines:] if new_lines else []
    return len(fresh), sum(row.get("est_usd", 0.0) for row in fresh)


def _sum_tokens(path: Path, new_lines: int) -> tuple[int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fresh = rows[-new_lines:] if new_lines else []
    return (sum(row.get("input_tokens", 0) for row in fresh), sum(row.get("output_tokens", 0) for row in fresh))


def default_reviewer(model: str, *, resolution: str = "high", processing: str = "static", attempts: int = 3, sleep=None):
    from .review_gemini import GeminiReviewer

    return GeminiReviewer(model=normalize_model(model), resolution=resolution, processing=processing,
                          attempts=attempts, sleep=sleep)
