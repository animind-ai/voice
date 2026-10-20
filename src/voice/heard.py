"""Whisper 听写结果的读法归一化：把 1/2、2分之1、x²、25%、-3、+3 这类阿拉伯数字和符号写法，展开成和稿子 `$...$` 公式一样的读法。

稿子经 zh-math-reading 朗读后是“二分之一”“x 的平方”“负三”“正三”，Whisper 却常把念出来的数学听写回数字和记号。
两边逐字比对就成了大段 replace，落在里面的书签只能线性插值。这里分三层：

1. 读法：每个数学表达式（如 “-25%”、“x+1/2”、“a^-2”）先换成等价的 LaTeX，再交给 zh_math_reading.normalize 读，
   和稿子里 `$...$` 的读法逐字相同；一元的正负号和二元的加减由 LaTeX 的上下文决定。
2. 数字：裸数字（如 “10”、“+3”）有两种读法（中文读法，或阿拉伯数字原样）。整句做一次格子上的对齐动态规划，
   同时决定每个数字用哪种读法，使整句与稿子的读法匹配最多；不是逐个贪心地定。
3. 时间：表达式里和读法逐字相同的字母或数字（如 “x”）保留它自己的词的时间段；插入的读法字（的平方、分之、负）
   分在它们前后两个锚点之间的时间里；没有锚点的表达式（如 “1/2”）的字平均分在整个表达式的时间段里。
   词的时间段按每个内容字的展开字数比例分（所以 “看1/2好” 中的 “好” 从 5/6 秒开始）。

表达式的边界（唯一的规则，expand 和 chars_from_words 共用）：相邻两个词里的字，只有两边都是字母或数字时才不连成一个表达式；
时间的长短不影响表达式的语义。
"""

from __future__ import annotations

import difflib
import re

from zh_math_reading import normalize

from . import textalign
from .textalign import HeardChar

_SUP = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"}
_BARE = re.compile(r"[+-]?\d+(?:\.\d+)?")  # 只有一个数（可带正负号）：有两种读法
_CNFRAC = re.compile(r"(\d+)分之(\d+)")  # “2分之1”：分母在前
# 数学表达式：一串数学字符；前面可以是“数分之数”
_RUN = re.compile(r"\d+分之\d+|[A-Za-z0-9.+\-*/=×÷%²³⁴⁵⁶⁷⁸⁹⁰¹^√]+")
_OPERAND = r"(?:[A-Za-z]|\d+(?:\.\d+)?)"
# 单独的运算符（两边是别的词）是二元的：读法同 “a+b” 里的运算符。zh_math_reading 对单独的 “+” 读作“正”（一元），这里不用。
_BINARY_OP = {"+": "加", "-": "减", "=": "等于", "×": "乘", "*": "乘", "÷": "除以", "≤": "小于等于", "≥": "大于等于"}

Option = tuple[str, ...]  # 一个单元的读法选项；单元有 1 个选项（固定读法）或 2 个（裸数字）


def _latex(run: str) -> str:
    """把 Whisper 写的数学表达式换成等价的 LaTeX（和稿子里 $...$ 的写法一样），供 zh_math_reading 读。"""
    t = re.sub(rf"({_OPERAND})([{''.join(_SUP)}]+)",
               lambda m: m.group(1) + "^{" + "".join(_SUP[c] for c in m.group(2)) + "}", run)
    t = re.sub(r"\^(-?\d+)", r"^{\1}", t)  # 记号写法：x^2、a^-2
    t = re.sub(rf"(?<![\w^{{])([+-]?)({_OPERAND})/({_OPERAND})", r"\1\\frac{\2}{\3}", t)  # 1/2、-1/2、x+1/2
    t = re.sub(r"(\d+(?:\.\d+)?)%", r"\1\\%", t)
    t = re.sub(r"√\s*(\d+(?:\.\d+)?|[A-Za-z])", r"\\sqrt{\1}", t)
    return t.replace("×", r"\times").replace("*", r"\times").replace("÷", r"\div").replace("≤", r"\le").replace("≥", r"\ge")


def _math_options(run: str) -> tuple[str, ...] | None:
    """一个数学表达式的读法选项；不是式子（只有符号）时返回 None。"""
    m = _CNFRAC.fullmatch(run)
    if m:
        return (normalize(rf"${{\frac{{{m.group(2)}}}{{{m.group(1)}}}}}$"),)
    if _BARE.fullmatch(run):
        return (normalize(f"${run}$"), run)  # 中文读法在前（平手时用它）
    if not re.search(r"[A-Za-z0-9]", run):
        return (_BINARY_OP[run],) if run in _BINARY_OP else None
    return (normalize(f"${_latex(run)}$"),)


def _wordlike(ch: str) -> bool:
    """字母、汉字、数字（不含上标数字）。相邻两个词的这两个字都是它时，表达式不能跨过这个边界。"""
    return ch.isalnum() and ch not in _SUP


def _chars(words: list[dict]) -> list[tuple[str, int]]:
    return [(c, wi) for wi, w in enumerate(words) for c in w["word"] if not c.isspace()]


def _blocks(chars: list[tuple[str, int]]) -> list[tuple[int, int]]:
    cuts = [0]
    for i in range(1, len(chars)):
        if chars[i][1] != chars[i - 1][1] and _wordlike(chars[i - 1][0]) and _wordlike(chars[i][0]):
            cuts.append(i)
    cuts.append(len(chars))
    return list(zip(cuts, cuts[1:]))


def _units(chars: list[tuple[str, int]]) -> list[tuple[int, int, Option]]:
    """切成单元 (起, 止, 读法选项)。数学表达式、裸数字是一个单元；其余每个字一个单元。"""
    text = "".join(ch for ch, _ in chars)
    units: list[tuple[int, int, Option]] = []
    for lo, hi in _blocks(chars):
        seg, pos = text[lo:hi], 0
        for m in _RUN.finditer(seg):
            units += [(lo + i, lo + i + 1, (seg[i],)) for i in range(pos, m.start())]
            try:
                opts = _math_options(m.group(0))
            except Exception:  # 读不了的写法（如不完整的式子）按字拆开
                opts = None
            if opts is None:
                units += [(lo + i, lo + i + 1, (seg[i],)) for i in range(m.start(), m.end())]
            else:
                units.append((lo + m.start(), lo + m.end(), opts))
            pos = m.end()
        units += [(lo + i, lo + i + 1, (seg[i],)) for i in range(pos, len(seg))]
    return units


def _choose(units: list[tuple[int, int, Option]], script: str) -> list[int]:
    """整句的格子动态规划：每个单元选一个读法，使整句的读法与稿子的读法（归一化后的内容字）的匹配字数最多。

    状态是（单元边界，稿子位置），沿着每个选项的字走：字相同就匹配（+1），不同就跳过听写的字，稿子的字也可以跳过。
    平手时选靠前的选项（中文读法）。
    """
    target = [textalign.normalize(c) for c in script if textalign.is_content(c)]
    n_target = len(target)
    neg = float("-inf")

    def skip(arr):  # 稿子的字可以不被听写：同一层里向后传递
        for q in range(n_target):
            if arr[q][0] > arr[q + 1][0]:
                arr[q + 1] = (arr[q][0], "s")
        return arr

    junctions = [skip([(0.0, "start")] + [(neg, None)] * n_target)]
    chains: list[list[list]] = []
    for _, _, opts in units:
        per_option = []
        for opt in opts:
            heard = [textalign.normalize(c) for c in opt if textalign.is_content(c)]
            layers = [junctions[-1]]
            for ch in heard:
                prev, new = layers[-1], [(neg, None)] * (n_target + 1)
                for q in range(n_target + 1):
                    s = prev[q][0]
                    if s == neg:
                        continue
                    if s > new[q][0]:
                        new[q] = (s, "h")  # 听写的字没有对上稿子
                    if q < n_target and target[q] == ch and s + 1 > new[q + 1][0]:
                        new[q + 1] = (s + 1, "m")  # 对上了
                layers.append(skip(new))
            per_option.append(layers)
        new = [(neg, None)] * (n_target + 1)
        for oi, layers in enumerate(per_option):
            for q, (s, _) in enumerate(layers[-1]):
                if s > new[q][0]:
                    new[q] = (s, ("opt", oi))
        junctions.append(skip(new))
        chains.append(per_option)

    choice = [0] * len(units)
    k = len(units)
    q = max(range(n_target + 1), key=lambda j: junctions[k][j][0])
    while k > 0:
        kind = junctions[k][q][1]
        while kind == "s":
            q -= 1
            kind = junctions[k][q][1]
        oi = kind[1]
        choice[k - 1] = oi
        layers = chains[k - 1][oi]
        layer = len(layers) - 1
        while layer > 0:
            kind = layers[layer][q][1]
            if kind == "s":
                q -= 1
            elif kind == "h":
                layer -= 1
            else:  # "m"
                layer -= 1
                q -= 1
        k -= 1
    return choice


def _intervals(words: list[dict], chars, units, choice) -> list[tuple[float, float]]:
    """每个字的时间段。权重 = 它所在单元的展开内容字数 ÷ 单元里的内容字数，词的时间段按权重分给它的内容字。"""
    weight = [0.0] * len(chars)
    for (a, b, opts), c in zip(units, choice):
        raw = [i for i in range(a, b) if textalign.is_content(chars[i][0])]
        if raw:
            expanded = sum(1 for ch in opts[c] if textalign.is_content(ch))
            for i in raw:
                weight[i] = expanded / len(raw)
    out: list[tuple[float, float]] = [(0.0, 0.0)] * len(chars)
    pos = 0
    for w in words:
        idxs = list(range(pos, pos + sum(1 for c in w["word"] if not c.isspace())))
        pos += len(idxs)
        if not idxs:
            continue
        s, e = w["start"], w["end"]
        content = [i for i in idxs if textalign.is_content(chars[i][0])]
        if not content:  # 只有符号的词（如单独的 “%”）：整段时间给它的每个字
            for i in idxs:
                out[i] = (s, e)
            continue
        total = sum(weight[i] for i in content)
        cum, prev_end = 0.0, s
        for i in idxs:
            if textalign.is_content(chars[i][0]):
                lo = s + (e - s) * cum / total if total > 0 else s
                cum += weight[i]
                hi = s + (e - s) * cum / total if total > 0 else s
                out[i] = (lo, hi)
                prev_end = hi
            else:
                out[i] = (prev_end, prev_end)
    return out


def _emit(units, choice, chars, ivs) -> list[HeardChar]:
    """每个单元的内容字：和原字相同的字（锚点）保留它自己的时间段；其余的字分在锚点之间（或整个单元的时间段里）。"""
    out: list[HeardChar] = []
    for (a, b, opts), c in zip(units, choice):
        reading = opts[c]
        reading_chars = [ch for ch in reading if textalign.is_content(ch)]
        if not reading_chars:
            continue
        cand = [i for i in range(a, b) if _wordlike(chars[i][0])]
        matcher = difflib.SequenceMatcher(None, [chars[i][0] for i in cand], reading_chars, autojunk=False)
        anchor: dict[int, int] = {}  # 读法里的第几个字 → 原字的序号
        for blk in matcher.get_matching_blocks():
            for k in range(blk.size):
                anchor[blk.b + k] = cand[blk.a + k]
        nonzero = [ivs[i] for i in range(a, b) if ivs[i][1] > ivs[i][0]]
        if not nonzero:
            continue
        lo, hi = min(s for s, _ in nonzero), max(e for _, e in nonzero)
        r = 0
        while r < len(reading_chars):
            if r in anchor:
                s, e = ivs[anchor[r]]
                if e > s:
                    out.append(HeardChar(reading_chars[r], s, e))
                r += 1
                continue
            g0 = r
            while r < len(reading_chars) and r not in anchor:
                r += 1
            left = ivs[anchor[g0 - 1]][1] if g0 > 0 else lo
            right = ivs[anchor[r]][0] if r < len(reading_chars) else hi
            n = r - g0
            if right > left:
                out += [HeardChar(reading_chars[g0 + k], left + (right - left) * k / n, left + (right - left) * (k + 1) / n)
                        for k in range(n)]
    return out


def _resolve(words: list[dict], script: str):
    chars = _chars(words)
    units = _units(chars)
    choice = _choose(units, script)
    return chars, units, choice


def expand_words(words: list[dict], lang: str = "zh", script: str = "") -> str:
    """听写词展开成读法文本（和稿子的读法比对用）。英文不处理。"""
    if lang != "zh":
        return "".join(w["word"] for w in words)
    _, units, choice = _resolve(words, script)
    return "".join(opts[c] for (_, _, opts), c in zip(units, choice))


def expand(text: str, lang: str = "zh", script: str = "") -> str:
    """只有文本、没有词时间的版本：空白分开的是不同的词，没有空白的是同一个词。英文原样返回。"""
    if lang != "zh":
        return text
    return expand_words([{"word": tok, "start": 0.0, "end": 0.0} for tok in text.split()], lang, script)


def chars_from_words(words: list[dict], lang: str = "zh", script: str = "") -> list[HeardChar]:
    """Whisper 的词摊成逐字，每个字带时间段（规则见模块说明）。零时长的字借不到时间，不输出（书签插值，不误标为 matched）。"""
    if lang != "zh":
        return textalign.chars_from_words(words)
    ws = [w for w in words if any(not c.isspace() for c in w["word"])]
    chars, units, choice = _resolve(ws, script)
    ivs = _intervals(ws, chars, units, choice)
    return _emit(units, choice, chars, ivs)
