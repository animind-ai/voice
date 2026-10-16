"""纯文本层的对齐：把稿子的字和听写出来的字配对，用于求书签时刻和判断念得对不对。

不依赖任何语音模型，方便单测。
"""

from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass

DIGITS = str.maketrans("0123456789", "〇一二三四五六七八九")


def is_content(ch: str) -> bool:
    """参与对齐的字：去掉标点、空白、符号。"""
    if ch.isspace():
        return False
    cat = unicodedata.category(ch)
    return not (cat.startswith("P") or cat.startswith("S") or cat.startswith("Z"))


def normalize(ch: str) -> str:
    return ch.lower().translate(DIGITS)


@dataclass
class HeardChar:
    ch: str
    start: float
    end: float


def content_index(text: str, offset: int) -> int:
    """text[:offset] 里有几个参与对齐的字 —— 即书签落在第几个字前面。"""
    return sum(1 for ch in text[:offset] if is_content(ch))


@dataclass
class Alignment:
    script: list[str]
    heard: list[HeardChar]
    mapping: dict[int, int]  # 稿子第 i 个字 → 听写第 j 个字
    ratio: float  # 匹配上的字占稿子字数的比例

    def time_at(self, i: int, total: float) -> tuple[float, str]:
        """稿子第 i 个字开始的时刻，以及怎么得到的（matched / interpolated / estimated）。"""
        n = len(self.script)
        if i >= n:
            last = max((self.heard[j].end for j in self.mapping.values()), default=total)
            return last, "matched" if self.mapping else "estimated"
        if i in self.mapping:
            return self.heard[self.mapping[i]].start, "matched"
        before = max((k for k in self.mapping if k < i), default=None)
        after = min((k for k in self.mapping if k > i), default=None)
        if before is not None and after is not None:
            t0 = self.heard[self.mapping[before]].end
            t1 = self.heard[self.mapping[after]].start
            frac = (i - before) / (after - before)
            return t0 + (t1 - t0) * frac, "interpolated"
        if before is not None:
            t0 = self.heard[self.mapping[before]].end
            return t0 + (total - t0) * (i - before) / max(1, n - before), "interpolated"
        if after is not None:
            t1 = self.heard[self.mapping[after]].start
            return t1 * i / max(1, after), "interpolated"
        return total * i / max(1, n), "estimated"


def align(script_text: str, heard: list[HeardChar]) -> Alignment:
    script = [normalize(ch) for ch in script_text if is_content(ch)]
    heard_c = [h for h in heard if is_content(h.ch)]
    matcher = difflib.SequenceMatcher(None, script, [normalize(h.ch) for h in heard_c], autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
    ratio = len(mapping) / max(1, len(script))
    return Alignment(script, heard_c, mapping, ratio)


def chars_from_words(words: list[dict]) -> list[HeardChar]:
    """Whisper 的词（对中文多为单字，偶尔带标点或几个字）摊成逐字，时间在词内均分。"""
    out: list[HeardChar] = []
    for w in words:
        text = w["word"].strip()
        content = [ch for ch in text if is_content(ch)]
        if not content:
            continue
        span = (w["end"] - w["start"]) / len(content)
        for k, ch in enumerate(content):
            out.append(HeardChar(ch, w["start"] + k * span, w["start"] + (k + 1) * span))
    return out


@dataclass
class Diff:
    missing: list[str]  # 稿子里有、没听到的片段
    extra: list[str]  # 听到了、稿子里没有的片段


def diff(script_text: str, heard_text: str) -> Diff:
    a = [normalize(c) for c in script_text if is_content(c)]
    b = [normalize(c) for c in heard_text if is_content(c)]
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    missing, extra = [], []
    for op, i1, i2, j1, j2 in m.get_opcodes():
        if op in ("delete", "replace") and i2 > i1:
            missing.append("".join(a[i1:i2]))
        if op in ("insert", "replace") and j2 > j1:
            extra.append("".join(b[j1:j2]))
    return Diff(missing, extra)
