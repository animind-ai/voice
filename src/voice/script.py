"""从 scene-spec 文档取出要念的句子：去掉书签后的读法文本，以及每个书签在读法文本里的字符位置。

读法文本先经 zh-math-reading 的 normalize()：展开 `$...$` 公式、改写容易被 TTS 吞字的命名句。
normalize 保留书签，所以书签位置在展开之后再算，自然指向展开后的文字。
不含公式、也没有要改写的句式时 normalize 原样返回，已有的音频缓存（按读法文本做键）不受影响。

英文版（scene_spec.localize(doc, "en") 之后 style.lang 是 en）不过 normalize：英文稿本来就把公式写成词。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from zh_math_reading import lint, normalize

_BOOKMARK = re.compile(r"\{@([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class Sentence:
    beat: str
    index: int
    id: str | None
    pace: str
    text: str  # 原文（含书签，可能含 $...$ 公式）
    spoken: str  # normalize 之后、去掉书签的读法文本（送给 TTS 的就是它）
    marks: dict[str, int] = field(default_factory=dict)  # 书签 → spoken 里的字符位置
    lang: str = "zh"

    @property
    def key(self) -> str:
        """timing.json 里引用这句的键：有 id 用 id，否则用下标。"""
        return self.id if self.id else str(self.index)

    def warnings(self) -> list[str]:
        """zh-math-reading 对读法文本的风险提示（冒号后的字母、长句、多音字……）；英文版没有。"""
        return lint(self.spoken) if self.lang == "zh" else []


@dataclass
class Beat:
    id: str
    sentences: list[Sentence]


def split_marks(text: str) -> tuple[str, dict[str, int]]:
    spoken, marks, pos = [], {}, 0
    for m in _BOOKMARK.finditer(text):
        spoken.append(text[pos : m.start()])
        marks[m.group(1)] = sum(len(s) for s in spoken)
        pos = m.end()
    spoken.append(text[pos:])
    return "".join(spoken), marks


def _tidy(spoken: str, marks: dict[str, int]) -> tuple[str, dict[str, int]]:
    """去掉书签留下的重复空格和首尾空格，书签位置跟着挪。"""
    keep = [k for k, ch in enumerate(spoken)
            if not (ch == " " and (k == 0 or spoken[k - 1] == " " or k == len(spoken) - 1))]
    text = "".join(spoken[k] for k in keep)
    return text, {name: sum(1 for k in keep if k < pos) for name, pos in marks.items()}


def lang(doc: dict) -> str:
    return doc.get("style", {}).get("lang", "zh")


def beats(doc: dict) -> list[Beat]:
    out, language = [], lang(doc)
    for beat in doc["beats"]:
        sentences = []
        for i, s in enumerate(beat.get("say", [])):
            text = normalize(s["text"]) if language == "zh" else " ".join(s["text"].split())
            spoken, marks = split_marks(text)
            if language != "zh":
                spoken, marks = _tidy(spoken, marks)
            sentences.append(Sentence(beat["id"], i, s.get("id"), s["pace"], s["text"], spoken, marks, language))
        out.append(Beat(beat["id"], sentences))
    return out
