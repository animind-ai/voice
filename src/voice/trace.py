"""把对齐警告和审片问题追溯回场景文件：beats[i].say[j] 在 .scene.yaml 的第几行。

行号来自 yaml.compose（SafeLoader）的节点起点。scene_spec.loader 只在出错时报位置，所以在这里另读一遍；
文件已经被 scene_spec 的受限读取器读过、合法，SafeLoader 组出来的节点结构和它一样。
句子的行号指向这句要念的文本：中文版是 text 的值，英文版（--lang en）是 en 的值；没有就指向这一项本身。

审片用的整集时间轴（timeline）：
- 首选 kit 写在成片旁边的 <成片>.chapters.json：每个 beat 的 start/end 就是成片里的起止（各段首尾相接），
  steps 是墨线后端的子步定格（成片里定住 hold 秒，旁白同步插入静音），把 timing.json 里的句子时刻换到成片上要加上
  它之前的定格；暂停（pause）的 hold 在段尾，落在最后一句之后，自然只对到 beat。
- 没有 chapters.json 时，只有每个 beat 都有 timing.json 才估：beat 长度按 kit 的下限
  （旁白长 + 0.28 秒 + pause.hold，至少 1 秒）累加。动画比旁白长时会偏早，所以 sources 里标 estimated。
- 都没有就不追溯，问题保持原样。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # scene-spec 的依赖

TAIL = 0.28  # kit 给每段旁白后面留的尾巴（render.py / moxian_backend.py 的 wav_seconds + 0.28）


@dataclass
class SayRef:
    beat_index: int
    beat_id: str
    index: int
    key: str  # 有 id 用 id，否则用下标（和 timing.json、script.Sentence.key 一样）
    line: int | None = None  # 1 起
    column: int | None = None

    @property
    def path(self) -> str:
        return f"beats[{self.beat_index}].say[{self.index}]"


@dataclass
class BeatRef:
    index: int
    id: str
    line: int | None = None
    column: int | None = None
    says: list[SayRef] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"beats[{self.index}]"


def _get(node, name: str):
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode) and key.value == name:
                return value
    return None


def _pos(node) -> tuple[int | None, int | None]:
    return (node.start_mark.line + 1, node.start_mark.column + 1) if node is not None else (None, None)


def scene_refs(doc: dict, source: str | Path | None = None, lang: str = "zh") -> dict[str, BeatRef]:
    """beat id → BeatRef（带每句的 SayRef）。source 是场景文件路径或文本；读不到行号时 line 为 None。"""
    root = None
    if source is not None:
        text = Path(source).read_text(encoding="utf-8") if isinstance(source, Path) else source
        try:
            root = yaml.compose(text, Loader=yaml.SafeLoader)
        except yaml.YAMLError:
            root = None
    beat_nodes = _get(root, "beats")
    beat_nodes = beat_nodes.value if isinstance(beat_nodes, yaml.SequenceNode) else []
    refs = {}
    for i, beat in enumerate(doc.get("beats", [])):
        node = beat_nodes[i] if i < len(beat_nodes) else None
        ref = BeatRef(i, beat["id"], *_pos(_get(node, "id") or node))
        say_nodes = _get(node, "say")
        say_nodes = say_nodes.value if isinstance(say_nodes, yaml.SequenceNode) else []
        for j, s in enumerate(beat.get("say", [])):
            item = say_nodes[j] if j < len(say_nodes) else None
            target = (_get(item, "en") if lang == "en" else None) or _get(item, "text") or item
            ref.says.append(SayRef(i, beat["id"], j, s.get("id") or str(j), *_pos(target)))
        refs[beat["id"]] = ref
    return refs


def describe(ref: SayRef | BeatRef, start: float | None = None, end: float | None = None, digits: int = 3) -> str:
    """beats[2].say[1] (line 37, t=12.300-15.800)；没有行号就省掉 line，没有时间就省掉 t。"""
    bits = []
    if ref.line is not None:
        bits.append(f"line {ref.line}")
    if start is not None:
        bits.append(f"t={start:.{digits}f}-{(start if end is None else end):.{digits}f}")
    return f"{ref.path} ({', '.join(bits)})" if bits else ref.path


# ---------- 整集时间轴 ----------


@dataclass
class Span:
    ref: SayRef | BeatRef
    start: float
    end: float


@dataclass
class Timeline:
    source: str  # chapters / estimated
    beats: list[Span]
    sentences: list[Span]  # 成片时间轴上的每句旁白

    def locate(self, start: float, end: float) -> tuple[Span | None, Span | None]:
        """(beat, sentence)。和问题时间重叠最多的那一句；落在两句之间（或 pause、没有旁白的 beat）只给 beat；
        不在任何 beat 里是 (None, None)。时间是一个点时按包含算。"""
        def best(spans: list[Span]) -> Span | None:
            if end > start:
                scored = [(min(end, s.end) - max(start, s.start), s) for s in spans]
                scored = [(o, s) for o, s in scored if o > 1e-6]
                return max(scored, key=lambda x: x[0])[1] if scored else None
            return next((s for s in spans if s.start <= start <= s.end), None)

        beat = best(self.beats)
        if beat is None:
            return None, None
        sentence = best([s for s in self.sentences if s.ref.beat_index == beat.ref.index])
        return beat, sentence

    def span_of(self, ref: SayRef) -> Span | None:
        """这句旁白在成片上的时间窗；没有音频（没有 timing.json）就是 None。"""
        return next((s for s in self.sentences if s.ref is ref), None)


def _load_timing(audio: Path | None, beat_id: str) -> dict | None:
    path = audio / f"{beat_id}.timing.json" if audio is not None else None
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ms(seconds: float) -> int:
    """时刻都按整数毫秒算：chapters.json 里的绝对时刻相减会带浮点误差（12.3 - 11.0 = 1.3000000000000007），
    直接比较会把正好落在定格点上的句子漏掉整个 hold。"""
    return round(float(seconds) * 1000)


def _to_video(t: float, freezes: list[tuple[int, int]]) -> float:
    """beat 内的旁白时刻 → 成片里的 beat 内时刻：加上它之前（含同一时刻）所有定格的 hold（同 kit segments.to_video）。"""
    t_ms = _ms(t)
    return (t_ms + sum(hold for at, hold in freezes if at <= t_ms)) / 1000


def _freezes(chapter: dict) -> list[tuple[int, int]]:
    """chapters.json 的 steps（成片时刻 time、hold）→ 定格在旁白时钟上的位置 (at, hold)，单位毫秒。"""
    out, before, start = [], 0, _ms(chapter["start"])
    for step in sorted(chapter.get("steps") or [], key=lambda s: s["time"]):
        hold = _ms(step["hold"])
        out.append((_ms(step["time"]) - start - before, hold))
        before += hold
    return out


def build_timeline(refs: dict[str, BeatRef], audio: Path | None, chapters: Path | None, doc: dict) -> Timeline | None:
    beats, sentences = [], []
    if chapters is not None and chapters.exists():
        data = json.loads(chapters.read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in data.get("chapters", [])}
        for ref in refs.values():
            chapter = by_id.get(ref.id)
            if chapter is None:
                continue
            beats.append(Span(ref, float(chapter["start"]), float(chapter["end"])))
            timing = _load_timing(audio, ref.id)
            cues = (timing or {}).get("cues") or []
            if len(cues) != len(ref.says):
                continue  # 没有旁白音频，或稿子改过、和音频对不上：只对到 beat
            freezes = _freezes(chapter)
            for say, cue in zip(ref.says, cues):
                sentences.append(Span(say, chapter["start"] + _to_video(cue["start"], freezes),
                                      chapter["start"] + _to_video(cue["end"], freezes)))
        return Timeline("chapters", beats, sentences) if beats else None
    timings = {ref.id: _load_timing(audio, ref.id) for ref in refs.values()}
    if not refs or any(t is None for t in timings.values()):
        return None
    cursor = 0.0
    for beat, ref in zip(doc.get("beats", []), refs.values()):
        timing = timings[ref.id]
        cues = timing.get("cues") or []
        hold = float(beat["pause"].get("hold", 2)) if "pause" in beat else 0.0
        length = max(float(timing.get("total", 0.0)) + TAIL + hold, 1.0)
        beats.append(Span(ref, cursor, cursor + length))
        if len(cues) == len(ref.says):
            sentences += [Span(say, cursor + cue["start"], cursor + cue["end"]) for say, cue in zip(ref.says, cues)]
        cursor += length
    return Timeline("estimated", beats, sentences)
