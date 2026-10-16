"""从 scene-spec 文档取出要念的句子：去掉书签后的读法文本，以及每个书签在读法文本里的字符位置。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BOOKMARK = re.compile(r"\{@([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class Sentence:
    beat: str
    index: int
    id: str | None
    pace: str
    text: str  # 原文（含书签）
    spoken: str  # 去掉书签后的读法文本
    marks: dict[str, int] = field(default_factory=dict)  # 书签 → spoken 里的字符位置

    @property
    def key(self) -> str:
        """timing.json 里引用这句的键：有 id 用 id，否则用下标。"""
        return self.id if self.id else str(self.index)


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


def beats(doc: dict) -> list[Beat]:
    out = []
    for beat in doc["beats"]:
        sentences = []
        for i, s in enumerate(beat.get("say", [])):
            spoken, marks = split_marks(s["text"])
            sentences.append(Sentence(beat["id"], i, s.get("id"), s["pace"], s["text"], spoken, marks))
        out.append(Beat(beat["id"], sentences))
    return out
