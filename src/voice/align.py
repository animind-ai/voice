"""书签对齐：对每个 beat，按 timing.json 的句子起止把 beat 音频切开，用 Whisper 听写出逐字时间戳，
再和稿子逐字配对，得到每个书签的真实时刻；顺带给出每句念得对不对（回听检查）。

结果写回 <beat>.timing.json：
    "marks": {"s1": {"d0": 3.21, ...}, ...}      # 绝对时刻（相对 beat 音频开头）
    "check": {"s1": {"ratio": 0.97, "missing": [...], "extra": [...], "heard": "..."}}
    "words": {"s1": {"n": 42, "at": [[0, 0.31], [5, 0.62], ...]}}  # 稿子第 i 个参与对齐的字（词首）的时刻，n 是总字数；kit 用来给字幕切条定时
    "aligner": "mlx-community/whisper-large-v3-turbo"

听写后端按平台自动选：macOS 用 mlx-whisper（Apple 芯片），其他平台用 faster-whisper（CTranslate2，有 CUDA 就用 GPU）。
环境变量 ANIMIND_ALIGN_BACKEND=mlx|faster 或 CLI 的 --backend 可以强制指定。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import heard, policy, textalign
from .audio import slice_16k
from .script import Beat, Sentence
from .ttscache import write_text_atomic


class Inconsistent(ValueError):
    """这一段的 wav、timing.json 和发布说明对不上（policy.check_beat），不能对齐。"""


BACKENDS = ("mlx", "faster")
DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-turbo",
    "faster": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # 同一个 large-v3-turbo 的 CTranslate2 版本
}
PAD = 0.15  # 切片两侧多留一点，免得句首句尾的字被切掉


@dataclass
class SentenceResult:
    key: str
    ratio: float
    marks: dict[str, float] = field(default_factory=dict)
    how: dict[str, str] = field(default_factory=dict)  # 书签 → matched / interpolated / estimated
    heard: str = ""
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    confirmed: list[str] | None = None  # 复核后仍然缺的字；None 表示没复核
    words: list[list[float]] = field(default_factory=list)  # [[稿子里第几个字（词首）, 时刻], ...]，只记精确匹配上的
    content: int = 0  # 稿子里参与对齐的字数
    index: int = 0  # 这句在 beat 的 say 里的下标
    start: float = 0.0  # 这句在 beat 音频里的起止（timing.json 的 cue）
    end: float = 0.0

    @property
    def suspicious(self) -> bool:
        """第一遍听写对不上：匹配率低，或者漏了 / 多了两个字以上的片段。"""
        return self.ratio < 0.85 or any(len(s) >= 2 for s in self.missing + self.extra)

    @property
    def flagged(self) -> bool:
        """需要人去听：第一遍可疑，并且复核后仍有缺字（没有原始单句音频可复核时，按第一遍算）。"""
        if not self.suspicious:
            return False
        return bool(self.confirmed) if self.confirmed is not None else True


def default_backend() -> str:
    forced = os.environ.get("ANIMIND_ALIGN_BACKEND", "").strip().lower()
    if forced:
        if forced not in BACKENDS:
            raise ValueError(f"ANIMIND_ALIGN_BACKEND={forced!r}：只能是 {' / '.join(BACKENDS)}")
        return forced
    return "mlx" if sys.platform == "darwin" else "faster"


def transcribe_mlx(samples, prompt: str, model: str, language: str = "zh") -> tuple[str, list[dict]]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        samples, path_or_hf_repo=model, language=language, word_timestamps=True,
        initial_prompt=prompt or None, verbose=None, condition_on_previous_text=False,
        compression_ratio_threshold=1.8,  # 比默认 2.4 严，更早触发温度回退，压住重复死循环
    )
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    return result.get("text", "").strip(), words


_faster_models: dict = {}


def _faster_model(model: str):
    if model not in _faster_models:
        from faster_whisper import WhisperModel

        device = os.environ.get("ANIMIND_ALIGN_DEVICE")
        if not device:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        _faster_models[model] = WhisperModel(model, device=device, compute_type=compute)
    return _faster_models[model]


def transcribe_faster(samples, prompt: str, model: str, language: str = "zh") -> tuple[str, list[dict]]:
    """和 transcribe_mlx 同样的参数、同样的输出：全文 + [{"word", "start", "end"}, ...]。"""
    segments, _ = _faster_model(model).transcribe(
        samples, language=language, word_timestamps=True, initial_prompt=prompt or None,
        condition_on_previous_text=False, compression_ratio_threshold=1.8,
        beam_size=1,  # 贪心解码，和 mlx-whisper 一致
        vad_filter=False,  # 切片本来就是一句话；VAD 会把句首句尾的轻声字切掉
    )
    text, words = [], []
    for seg in segments:
        text.append(seg.text)
        words += [{"word": w.word, "start": w.start, "end": w.end, "probability": w.probability} for w in seg.words or []]
    return "".join(text).strip(), words


TRANSCRIBERS = {"mlx": transcribe_mlx, "faster": transcribe_faster}


def transcribe(samples, prompt: str, model: str, language: str = "zh") -> tuple[str, list[dict]]:
    return TRANSCRIBERS[default_backend()](samples, prompt, model, language)


def _looks_hallucinated(spoken: str, heard_text: str, lang: str = "zh", words: list[dict] | None = None) -> bool:
    """Whisper 的重复死循环或乱码：听写比稿子长很多，或者和稿子几乎对不上。"""
    heard_text = _heard_text(heard_text, lang, spoken, words)
    a = sum(1 for c in spoken if textalign.is_content(c))
    b = sum(1 for c in heard_text if textalign.is_content(c))
    if b > 1.4 * a + 4:
        return True
    d = textalign.diff(spoken, heard_text)
    missing = sum(len(x) for x in d.missing)
    return missing > 0.4 * a


def _compare_text(text: str, words: list[dict] | None) -> str:
    """听写的全文，词与词之间加空格：和逐字时间用的是同一套词边界（heard 里空白分开的就是不同的词）。
    Whisper 的逐字时间戳有时没覆盖到句尾，全文比时间戳多出来的部分原样接在后面（没有时间，但要算进缺字检查）。"""
    if not words:
        return text
    tokens = [w["word"].strip() for w in words]
    joined = "".join(tokens)
    flat = text.replace(" ", "")
    if not flat.startswith(joined):
        return text
    tail = flat[len(joined):]
    return " ".join(tokens + ([tail] if tail else []))


def _heard_text(text: str, lang: str, script: str, words: list[dict] | None = None) -> str:
    """听写和稿子比对前的归一化：英文把数字拼成单词；中文把数字和记号展开成读法（数字按稿子的读法定）。"""
    if lang == "en":
        return textalign.spell_numbers(text)
    return heard.expand(_compare_text(text, words), lang, script)


def _transcribe_all(wav: Path, s0: float, s1: float, spoken: str, model: str, transcriber,
                    language: str = "zh") -> tuple[str, list[dict]]:
    """听写一句；逐字时间戳没覆盖到句尾时（Whisper 偶尔在半个多字节字符处断掉），从断点往后补听，最多补两次。

    听写像是幻觉（重复死循环、乱码）时，去掉提示词再听一遍，取和稿子更接近的那次。
    """
    samples = slice_16k(wav, s0, s1)
    heard_text, words = transcriber(samples, spoken, model, language)
    if _looks_hallucinated(spoken, heard_text, language, words):
        retry_text, retry_words = transcriber(samples, "", model, language)
        if textalign.align(spoken, heard.chars_from_words(retry_words, language, spoken)).ratio > \
                textalign.align(spoken, heard.chars_from_words(words, language, spoken)).ratio:
            heard_text, words = retry_text, retry_words
    for _ in range(2):
        last = max((w["end"] for w in words), default=0.0)
        if s0 + last >= s1 - 0.6:
            break
        covered = len(heard.chars_from_words(words, language, spoken))  # 展开之后的字数，和稿子的字数对得上
        rest = spoken[_offset_after(spoken, covered):]
        if not rest.strip():
            break
        cut = max(0.0, last - 0.1)
        _, more = transcriber(slice_16k(wav, s0 + cut, s1), rest, model, language)
        if not more:
            break
        words = words + [{**w, "start": w["start"] + cut, "end": w["end"] + cut} for w in more]
    return heard_text, words


def _offset_after(text: str, n: int) -> int:
    """text 里第 n 个参与对齐的字之后的位置。"""
    seen = 0
    for i, ch in enumerate(text):
        if textalign.is_content(ch):
            seen += 1
            if seen == n:
                return i + 1
    return len(text)


def align_sentence(sentence: Sentence, wav: Path, start: float, end: float, model: str, transcriber=transcribe) -> SentenceResult:
    s0 = max(0.0, start - PAD)
    heard_text, words = _transcribe_all(wav, s0, end + PAD, sentence.spoken, model, transcriber, sentence.lang)
    if sentence.lang == "en":  # Whisper 常把念出来的 two 写成 2：先拼回单词再和稿子配对
        heard_text = textalign.spell_numbers(heard_text)
        words = [{**w, "word": textalign.spell_numbers(w["word"])} for w in words]
    chars = heard.chars_from_words(words, sentence.lang, sentence.spoken)
    al = textalign.align(sentence.spoken, chars)
    res = SentenceResult(sentence.key, round(al.ratio, 3), heard=heard_text)
    for name, offset in sentence.marks.items():
        i = textalign.content_index(sentence.spoken, offset)
        t, how = al.time_at(i, end + PAD - s0)
        res.marks[name] = round(s0 + t, 3)
        res.how[name] = how
    starts, k, prev = [], 0, False
    for ch in sentence.spoken:  # 词首：前一个字符不参与对齐（空格、标点）的字
        if textalign.is_content(ch):
            if not prev:
                starts.append(k)
            k += 1
        prev = textalign.is_content(ch)
    res.words = [[i, round(s0 + al.heard[al.mapping[i]].start, 3)] for i in starts if i in al.mapping]
    res.content = len(al.script)
    d = textalign.diff(sentence.spoken, _heard_text(heard_text, sentence.lang, sentence.spoken, words))
    res.missing, res.extra = d.missing, d.extra
    return res


def confirm(sentence: Sentence, raw_wav: Path, model: str, transcriber=transcribe) -> list[str]:
    """在未变速的原始单句音频上，不带提示词、带提示词各听一次；两次都没听到的字才算确认缺失。

    第一遍是在拼好（可能变速）的 beat 音频上、带稿子作提示词听的，两种条件都会让 Whisper 听错：
    变速让字更难辨认，提示词会把它往稿子上带，也可能带偏。
    """
    samples = slice_16k(raw_wav, 0.0, 600.0)
    heard_plain, words_plain = transcriber(samples, "", model, sentence.lang)
    heard_prompted, words_prompted = transcriber(samples, sentence.spoken, model, sentence.lang)
    heard_plain = _heard_text(heard_plain, sentence.lang, sentence.spoken, words_plain)
    heard_prompted = _heard_text(heard_prompted, sentence.lang, sentence.spoken, words_prompted)
    gone_a = set("".join(textalign.diff(sentence.spoken, heard_plain).missing))
    gone_b = set("".join(textalign.diff(sentence.spoken, heard_prompted).missing))
    both = gone_a & gone_b
    return [m for m in textalign.diff(sentence.spoken, heard_prompted).missing if set(m) & both]


def align_beat(beat: Beat, audio_dir: Path, model: str | None = None, transcriber=transcribe, log=print) -> list[SentenceResult]:
    model = model or DEFAULT_MODELS[default_backend()]
    timing_path = audio_dir / f"{beat.id}.timing.json"
    wav = audio_dir / f"{beat.id}.wav"
    if not beat.sentences:
        return []
    if not timing_path.exists() or not wav.exists():
        raise FileNotFoundError(f"{beat.id}：缺少 {wav.name} 或 {timing_path.name}，先运行 synth")
    check = policy.check_beat(audio_dir, beat.id)
    if not check.clean:  # 按对不上 wav 的 cues 去切音频，书签时刻全是错的
        raise Inconsistent(f"{beat.id}：{check.describe()}，不对齐。"
                           f"先运行 synth（能修的会按发布说明修好；修不好的要重合成：synth --only {beat.id}）")
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    cues = timing["cues"]
    if len(cues) != len(beat.sentences):
        raise ValueError(f"{timing_path.name} 有 {len(cues)} 句，scene 里 {beat.id} 有 {len(beat.sentences)} 句；稿子改过就要重新 synth")
    results = []
    for sentence, cue in zip(beat.sentences, cues):
        r = align_sentence(sentence, wav, cue["start"], cue["end"], model, transcriber)
        r.index, r.start, r.end = sentence.index, cue["start"], cue["end"]
        raw = audio_dir / "lines" / f"{beat.id}-{sentence.index}.wav"
        if r.suspicious and raw.exists():
            r.confirmed = confirm(sentence, raw, model, transcriber)
        results.append(r)
    timing["marks"] = {r.key: r.marks for r in results if r.marks}
    timing["check"] = {r.key: {"ratio": r.ratio, "missing": r.missing, "extra": r.extra, "heard": r.heard,
                               "confirmed": r.confirmed} for r in results}
    timing["words"] = {r.key: {"n": r.content, "at": r.words} for r in results if r.words}
    timing["how"] = {r.key: r.how for r in results if r.how}  # 每个书签：matched / interpolated / estimated
    timing["aligner"] = model
    write_text_atomic(timing_path, json.dumps(timing, ensure_ascii=False, indent=1))
    flagged = [r.key for r in results if r.flagged]
    est = sum(1 for r in results for h in r.how.values() if h != "matched")
    log(f"{beat.id}: {len(results)} 句，书签 {sum(len(r.marks) for r in results)} 个（精确匹配率 {match_rate(results):.0%}，非精确匹配 {est}），需要回听 {flagged or '无'}")
    return results


def match_rate(results: list[SentenceResult]) -> float:
    """书签里精确匹配（matched）的比例；没有书签时为 1。"""
    hows = [h for r in results for h in r.how.values()]
    return sum(h == "matched" for h in hows) / len(hows) if hows else 1.0


def warnings(results: list[SentenceResult], refs: dict | None, beat_id: str) -> list[str]:
    """要人看的提示，每条带场景文件里的位置：beats[2].say[1] (line 37, t=12.300-15.800)。t 是 beat 音频里的时刻。

    书签不是逐字配上的（interpolated / estimated）各一条；需要回听的句子（漏念、多念、像幻觉）一条。
    refs 是 trace.scene_refs 的结果；没有就只写 beat 和句子的键。
    """
    from .trace import describe

    beat = (refs or {}).get(beat_id)
    out = []
    for r in results:
        ref = next((say for say in beat.says if say.index == r.index), None) if beat else None
        where = describe(ref, r.start, r.end) if ref else f"{beat_id}[{r.key}] (t={r.start:.3f}-{r.end:.3f})"
        for name, how in r.how.items():
            if how != "matched":
                out.append(f"{where}: 书签 {{@{name}}} {'插值' if how == 'interpolated' else '估算'}（{how}）在 {r.marks[name]:.3f}s，请核对")
        if r.flagged:
            sure = f"；复核仍缺 {r.confirmed}" if r.confirmed else "（没有原始单句音频，未复核）"
            out.append(f"{where}: 匹配 {r.ratio:.0%}；没听到 {r.missing}；多出 {r.extra}{sure}\n    听到：{r.heard}")
    return out
