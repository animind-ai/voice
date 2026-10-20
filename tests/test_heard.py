"""Whisper 听写的数字和记号展开（voice.heard），以及书签匹配率和时间的离线回归。不调用 Whisper，听写结果是手写的。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from zh_math_reading import normalize

from voice import align, heard, script, textalign

FIXTURE = Path(__file__).parent / "fixtures" / "align" / "formula_whisper.json"
DT = 0.2  # 回归集的真实时间线：每个读出来的字 0.2 秒


def _words(tokens, t0=0.1):
    """手写的听写词：每个词 0.2 秒加每个字 0.12 秒，依次排开。"""
    out, t = [], t0
    for tok in tokens:
        dur = 0.2 + 0.12 * len(tok)
        out.append({"word": tok, "start": round(t, 3), "end": round(t + dur, 3)})
        t += dur
    return out


def _truth_words(tokens, readings):
    """真实时间线：每个读出来的字 DT 秒，听写词的时间段就是它读出来的那些字的时间段。"""
    out, n = [], 0
    for tok, reading in zip(tokens, readings):
        k = len(reading)
        out.append({"word": tok, "start": round(n * DT, 6), "end": round((n + k) * DT, 6)})
        n += k
    return out


def _sentence(text):
    return script.beats({"beats": [{"id": "B", "say": [{"id": "s", "pace": "normal", "text": text}]}]})[0].sentences[0]


def _run(text, tokens, words=None, script_text=None):
    """对一句稿子跑 align_sentence，听写用假的 transcriber，返回结果。"""
    sentence = _sentence(text)
    words = words or _words(tokens)
    end = words[-1]["end"] - 0.3  # 句尾正好在最后一个词之后，不触发补听
    return align.align_sentence(sentence, Path("unused.wav"), 0.0, end, "m",
                                transcriber=lambda *a, **k: ("".join(tokens), words))


@pytest.fixture
def no_audio(monkeypatch):
    monkeypatch.setattr(align, "slice_16k", lambda *a: None)


def _strip(s):
    return "".join(s.split())


@pytest.mark.parametrize("whisper, script_text", [
    ("1/2", r"$\frac{1}{2}$"),
    ("3分之2", r"$\frac{2}{3}$"),
    ("-1/2", r"$-\frac{1}{2}$"),
    ("x²", "$x^2$"),
    ("x^2+y^2", r"$x^{2}+y^{2}$"),
    ("a^-2", r"$a^{-2}$"),
    ("25%", r"$25\%$"),
    ("-3", "$-3$"),
    ("3.5", "$3.5$"),
    ("√2", r"$\sqrt{2}$"),
    ("3x-2", "$3x-2$"),
    ("a=2", "$a=2$"),
    ("x-1/2", r"$x-\frac{1}{2}$"),  # 减号 + 分数：不能读成“负二分之一”
])
def test_expand_matches_script_reading(whisper, script_text):
    """听写的写法展开后，和稿子里同一个式子经 zh-math-reading 朗读的结果逐字相同（空格不计）。"""
    assert _strip(heard.expand(whisper)) == _strip(normalize(script_text))


def test_english_is_untouched():
    assert heard.expand("1/2 and x^2", lang="en") == "1/2 and x^2"
    heard_chars = heard.chars_from_words([{"word": " 2", "start": 0.0, "end": 0.2}], lang="en")
    assert [c.ch for c in heard_chars] == ["2"]


def test_word_time_is_split_over_expanded_chars():
    """“1/2” 的时间段 [1.0, 1.4] 分给“二分之一”的四个字，每个字 0.1 秒；标点不占时间。"""
    chars = heard.chars_from_words([{"word": "1/2", "start": 1.0, "end": 1.4}, {"word": "，", "start": 1.4, "end": 1.5}])
    assert [c.ch for c in chars] == ["二", "分", "之", "一"]
    assert [round(c.start, 6) for c in chars] == [1.0, 1.1, 1.2, 1.3]
    assert [round(c.end, 6) for c in chars] == [1.1, 1.2, 1.3, 1.4]


def test_plain_chinese_words_keep_old_timing():
    """没有数字和记号的词，逐字时间和原来的 chars_from_words 完全一样。"""
    words = [{"word": "下面", "start": 1.0, "end": 1.4}, {"word": "看，", "start": 1.4, "end": 1.9}]
    old = textalign.chars_from_words(words)
    assert [(c.ch, c.start, c.end) for c in heard.chars_from_words(words)] == [(c.ch, c.start, c.end) for c in old]


# ---------- 审查复现（每一条都是一个测试） ----------

def test_symbol_token_time_is_kept_when_split_from_number():
    """复现 1：'25' [0,1] 后面的 '%' [1,2]：符号的时间不能丢，“百分之二十五”要覆盖 [0,2]。"""
    chars = heard.chars_from_words([{"word": "25", "start": 0.0, "end": 1.0}, {"word": "%", "start": 1.0, "end": 2.0}])
    assert "".join(c.ch for c in chars) == "百分之二十五"
    assert chars[0].start == pytest.approx(0.0) and chars[-1].end == pytest.approx(2.0)
    assert chars[-1].start >= 1.0 - 1e-9  # 最后两个字落在 '%' 那个词的时间段里


def test_lone_symbol_word_keeps_its_span():
    """复现 1：单独的 '=' [1,2] 不能变成两个 [0,0] 的字，“等于”应占 [1,2]。"""
    chars = heard.chars_from_words([{"word": "=", "start": 1.0, "end": 2.0}])
    assert [c.ch for c in chars] == ["等", "于"]
    assert [(round(c.start, 6), round(c.end, 6)) for c in chars] == [(1.0, 1.5), (1.5, 2.0)]


def test_mixed_word_spreads_over_expanded_chars():
    """复现 1：'看1/2好' [0,1] 展开成六个字，“好”应从 5/6 秒开始，而不是 0.75。"""
    chars = heard.chars_from_words([{"word": "看1/2好", "start": 0.0, "end": 1.0}])
    assert "".join(c.ch for c in chars) == "看二分之一好"
    assert chars[-1].ch == "好" and chars[-1].start == pytest.approx(5 / 6)
    assert chars[-1].end == pytest.approx(1.0)


def test_binary_minus_before_fraction_is_not_a_negative_fraction():
    """复现 2：'x-1/2' 的减号是减号，读作“x减二分之一”，不是“x负二分之一”。"""
    assert _strip(heard.expand("x-1/2")) == "x减二分之一"
    assert "负" not in heard.expand("x-1/2")


def test_separate_numbers_are_not_merged_into_one():
    """复现 2：两个相邻的数字词“1”“2”不能合成“十二”，各自读自己的字，时间也各自分开。"""
    chars = heard.chars_from_words([{"word": "1", "start": 0.0, "end": 0.3}, {"word": "2", "start": 0.3, "end": 0.6}])
    assert [c.ch for c in chars] == ["一", "二"]
    assert [(round(c.start, 6), round(c.end, 6)) for c in chars] == [(0.0, 0.3), (0.3, 0.6)]


def test_numbers_separated_by_silence_are_not_merged():
    """复现 2：中间有停顿的“1”“2”也不合并，“2”的时间保留在 [0.8, 1.1]。"""
    chars = heard.chars_from_words([{"word": "1", "start": 0.0, "end": 0.3}, {"word": "2", "start": 0.8, "end": 1.1}])
    assert [c.ch for c in chars] == ["一", "二"]
    assert chars[1].start == pytest.approx(0.8) and chars[1].end == pytest.approx(1.1)


def test_bare_digits_kept_when_script_keeps_them(no_audio):
    """复现 3：稿子是“第10步”（normalize 不改裸数字），听写的 10 不能变成“十”，书签要 matched。"""
    res = _run("第{@a}10步", ["第", "10", "步"])
    assert res.how == {"a": "matched"} and res.ratio == 1.0
    res0 = _run("第{@a}0步", ["第", "0", "步"])
    assert res0.how == {"a": "matched"} and res0.ratio == 1.0


def test_bare_digits_read_as_numbers_when_script_says_them_in_words(no_audio):
    """复现 3 的另一面：稿子写的是“第十步”，听写的 10 就读成“十”，和稿子一致。"""
    assert heard.expand("10", "zh", "第十步") == "十"
    res = _run("第{@a}十步", ["第", "10", "步"])
    assert res.how == {"a": "matched"} and res.ratio == 1.0


def _has_formula(case) -> bool:
    """句子里有 $...$，或者听写里有数字和记号。"""
    return "$" in case["text"] or any(any(c.isdigit() or c in "²³√/%^=+-" for c in tok) for tok in case["heard"])


def test_regression_set_matched_rate_and_times(no_audio, monkeypatch):
    """回归集：书签的匹配标签和时间都要对。时间的真值是读出来的字按 DT 秒的时间线（和听写的展开无关）。"""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    after, truth_ok = [], []
    for case in cases:
        sentence = _sentence(case["text"])
        content = "".join(c for c in sentence.spoken if textalign.is_content(c))
        assert content == "".join(case["reading"]).replace(" ", ""), case["text"]  # 真值读法和稿子一致
        words = _truth_words(case["heard"], case["reading"])
        end = words[-1]["end"] - 0.3
        r = align.align_sentence(sentence, Path("unused.wav"), 0.0, end, "m",
                                 transcriber=lambda *a, _w=words, _t=case["heard"], **k: ("".join(_t), _w))
        after.append(r)
        for name, offset in sentence.marks.items():
            i = textalign.content_index(sentence.spoken, offset)
            if r.how[name] == "matched":
                truth_ok.append(abs(r.marks[name] - i * DT) < 0.002)
    all_hows = [h for r in after for h in r.how.values()]
    rate = sum(h == "matched" for h in all_hows) / len(all_hows)
    print(f"书签 {len(all_hows)} 个，精确匹配率 {rate:.0%}；匹配的书签时间与真值一致 {sum(truth_ok)}/{len(truth_ok)}")
    assert len(all_hows) >= 25
    assert rate >= 0.95
    assert truth_ok and all(truth_ok)  # 匹配的书签，时间必须落在真值上（误差 < 2 毫秒）


def test_regression_before_is_worse_on_formula_sentences(no_audio, monkeypatch):
    """同一批句子，关掉展开（改之前的行为）：公式句的字对齐率明显更低。"""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))
    formula = [c for c in cases if _has_formula(c)]

    def ratios(cs):
        out = []
        for c in cs:
            words = _truth_words(c["heard"], c["reading"])
            end = words[-1]["end"] - 0.3
            out.append(align.align_sentence(_sentence(c["text"]), Path("unused.wav"), 0.0, end, "m",
                                            transcriber=lambda *a, _w=words, _t=c["heard"], **k: ("".join(_t), _w)).ratio)
        return sum(out) / len(out)

    after = ratios(formula)
    monkeypatch.setattr(heard, "expand", lambda text, lang="zh", script="": text)
    monkeypatch.setattr(heard, "chars_from_words", lambda words, lang="zh", script="": textalign.chars_from_words(words))
    before = ratios(formula)
    print(f"含公式/数字的句子字对齐率：改之前 {before:.0%} → 改之后 {after:.0%}")
    assert after > before and after >= 0.95


def test_spaces_from_normalize_do_not_shift_bookmark_time(no_audio):
    """normalize 给公式读法加的空格不占时间：“$a b c d e$”读作五个字，书签在第六个字（甲）前面，真实时间 1.0 秒。"""
    text = "$a b c d e${@a}甲乙丙丁戊己庚辛壬癸"
    tokens = list("abcde甲乙丙丁戊己庚辛壬癸")
    words = [{"word": t, "start": round(i * DT, 6), "end": round((i + 1) * DT, 6)} for i, t in enumerate(tokens)]
    res = _run(text, tokens, words=words)
    assert res.how == {"a": "matched"}
    assert res.marks["a"] == pytest.approx(5 * DT, abs=0.002)


def test_match_rate_helper(no_audio):
    res = [_run("一{@a}二三", ["一", "二", "三"]), _run("四{@b}五", ["四五"])]
    assert align.match_rate(res) == 1.0
    assert align.match_rate([]) == 1.0


def test_prompt_is_the_normalized_reading(no_audio):
    """initial_prompt 是这句稿子的读法（公式已展开、书签已去掉），听写时带上它。"""
    seen = []
    sentence = _sentence("先看{@a}$\\frac{1}{2}$的意思。")

    def transcriber(samples, prompt, model, language="zh"):
        seen.append(prompt)
        return "先看1/2的意思", _words(["先看", "1/2", "的意思"])

    align.align_sentence(sentence, Path("unused.wav"), 0.0, 1.5, "m", transcriber=transcriber)
    assert seen[0] == "先看二分之一的意思。"


def test_mlx_and_faster_pass_normalized_prompt(monkeypatch):
    """两个后端都把提示词交给 Whisper 的 initial_prompt。"""
    seen = {}

    def fake_mlx(samples, path_or_hf_repo, **kw):
        seen["mlx"] = kw["initial_prompt"]
        return {"text": "", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", NS(transcribe=fake_mlx))
    align.transcribe_mlx(None, "二分之一的意思", "m")
    assert seen["mlx"] == "二分之一的意思"

    class FakeModel:
        def transcribe(self, samples, **kw):
            seen["faster"] = kw["initial_prompt"]
            return iter([]), None

    monkeypatch.setitem(align._faster_models, "m", FakeModel())
    align.transcribe_faster(None, "二分之一的意思", "m")
    assert seen["faster"] == "二分之一的意思"


# ---------- 第二轮审查复现（每一条都是一个测试；书签时间要断言内部的时刻） ----------

def test_same_sentence_bare_digit_and_formula_digit_each_get_their_own_reading(no_audio):
    """复现 1：同一句里裸数字 10 和公式里的 $10$ 各自按局部对齐选读法：裸数字保留 10，公式的 10 读作“十”。"""
    res = _run("第10步，取{@a}$10$。", ["第", "10", "步，取", "10"])
    assert res.how == {"a": "matched"} and res.ratio == 1.0


def test_formula_zero_is_not_confused_with_a_substring_of_bare_ten(no_audio):
    """复现 1：稿子是“第10步，取十。”，听写的 0 不能因为前面有 10 就被保留成数字。"""
    res = _run("第10步，取{@a}$0$。", ["第", "10", "步，取", "0"])
    assert res.how == {"a": "matched"} and res.ratio == 1.0


def test_binary_minus_survives_a_gap_between_words():
    """复现 2：x 和 -1/2 之间有 0.1 秒停顿，减号仍是减号（语义只看字符，不看停顿）。"""
    words = [{"word": "x", "start": 0.0, "end": 0.2}, {"word": "-1/2", "start": 0.3, "end": 1.3}]
    assert _strip(heard.expand_words(words)) == "x减二分之一"
    chars = heard.chars_from_words(words)
    assert "".join(c.ch for c in chars) == "x减二分之一"
    minus = next(c for c in chars if c.ch == "减")
    assert minus.end > minus.start and 0.2 <= minus.start < 0.5  # 减号占的时间在 x 和 “二” 之间，不是零时长


def test_whole_text_and_words_use_the_same_boundary_rule():
    """复现 2：1 和 2 是两个词（中间有空白或停顿），expand 和 chars_from_words 都不能把它们合成“十二”。"""
    assert heard.expand("1 2") == "一二"
    words = [{"word": "1", "start": 0.0, "end": 0.3}, {"word": "2", "start": 0.8, "end": 1.1}]
    assert heard.expand_words(words) == "一二"
    assert "".join(c.ch for c in heard.chars_from_words(words)) == "一二"
    assert heard.expand("12") == "十二"  # 没有空白的同一个词，仍然是一个数


def test_cross_word_expression_with_zero_duration_symbol_stays_inside_its_span():
    """复现 3：“25” [0,2] + “%” [2,2]：“%”没有时长，六个字全部落在 [0,2] 里，没有零时长的字。"""
    words = [{"word": "25", "start": 0.0, "end": 2.0}, {"word": "%", "start": 2.0, "end": 2.0}]
    chars = heard.chars_from_words(words)
    assert "".join(c.ch for c in chars) == "百分之二十五"
    assert all(c.end > c.start for c in chars)
    assert chars[0].start == pytest.approx(0.0) and chars[-1].end == pytest.approx(2.0)


def test_cross_word_expression_split_by_word_duration():
    """复现 3：“25” [0,1] + “%” [1,2]：两个词时长相同，所以前三个字（百分之）在 [0,1]，后三个字（二十五）在 [1,2]。"""
    chars = heard.chars_from_words([{"word": "25", "start": 0.0, "end": 1.0}, {"word": "%", "start": 1.0, "end": 2.0}])
    assert [round(c.start, 3) for c in chars] == [0.0, 0.333, 0.667, 1.0, 1.333, 1.667]
    assert [round(c.end, 3) for c in chars] == [0.333, 0.667, 1.0, 1.333, 1.667, 2.0]


def test_power_keeps_the_letter_interval_of_its_base_word():
    """复现 3：“x” [0,0.2] + “²” [0.2,0.8]：x 的字母仍然占 [0,0.2]，“的平方”占 “²” 的时间段。"""
    chars = heard.chars_from_words([{"word": "x", "start": 0.0, "end": 0.2}, {"word": "²", "start": 0.2, "end": 0.8}])
    assert [c.ch for c in chars] == ["x", "的", "平", "方"]
    assert (chars[0].start, chars[0].end) == (pytest.approx(0.0), pytest.approx(0.2))
    assert chars[1].start == pytest.approx(0.2)


def test_expression_inside_one_word_shares_the_word_by_expanded_count():
    """“看1/2好”[0,1]：展开后六个字平均分，“好”从 5/6 秒开始（不是按源码字符数）。"""
    chars = heard.chars_from_words([{"word": "看1/2好", "start": 0.0, "end": 1.0}])
    assert chars[-1].ch == "好" and chars[-1].start == pytest.approx(5 / 6)


def test_bookmarks_inside_a_cross_word_expression_get_real_times(no_audio):
    """复现 3（书签版）：“面积是$25\\%$，结束。”：书签 a 在“百”前面（1.0 秒），书签 b 在“结”前面（2.5 秒）。"""
    text = "面积是{@a}$25\\%${@b}，结束。"
    words = [{"word": "面积是", "start": 0.0, "end": 1.0}, {"word": "25", "start": 1.0, "end": 2.0},
             {"word": "%", "start": 2.0, "end": 2.5}, {"word": "，结束。", "start": 2.5, "end": 3.0}]
    res = _run(text, ["面积是", "25", "%", "，结束。"], words=words)
    assert res.how == {"a": "matched", "b": "matched"}
    assert res.marks["a"] == pytest.approx(1.0, abs=0.002)
    assert res.marks["b"] == pytest.approx(2.5, abs=0.002)


# ---------- 与 kit 共用的固定时间戳真值（tests/fixtures/align/timing_truth.json，和 kit 的是同一份） ----------

TIMING_TRUTH = json.loads((Path(__file__).parent / "fixtures" / "align" / "timing_truth.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", TIMING_TRUTH, ids=[c["text"][:12] for c in TIMING_TRUTH])
def test_bookmark_times_match_the_fixed_timestamps(case, no_audio):
    """Whisper 给出的每个字都是固定时间戳（标点也有固定时长：句中的停顿，一个字长）。align 出来的书签时刻，必须等于书签那个字的固定时刻（误差 < 2 毫秒）。
    kit 的无音频估算用的也是这份真值，两边都对得上，估算的比例才算可靠。"""
    chars = case["chars"]
    sentence = _sentence(case["text"])
    assert "".join(c for c in sentence.spoken if textalign.is_content(c)) == "".join(c for c, _, _ in chars if textalign.is_content(c))
    words = [{"word": c, "start": s, "end": e} for c, s, e in chars]
    end = chars[-1][2] - 0.3  # 句尾正好在最后一个字之后，不触发补听
    res = align.align_sentence(sentence, Path("unused.wav"), 0.0, end, "m",
                               transcriber=lambda *a, **k: ("".join(c for c, _, _ in chars), words))
    for name, index in case["marks"].items():
        assert res.how[name] == "matched", name
        assert res.marks[name] == pytest.approx(chars[index][1], abs=0.002), (name, res.marks[name], chars[index][1])


# ---------- 第三轮（最终轮）：读法与 zh-math-reading 逐字一致；数字联合求解；锚点时间 ----------

# (听写里的写法, 等价的 LaTeX（和稿子里 $...$ 的写法一样）, 同一个写法拆成几个听写词)
FORMS = [
    ("+3", "+3", ["+", "3"]),
    ("-3", "-3", ["-", "3"]),
    ("-25%", r"-25\%", ["-", "25", "%"]),
    ("+1/2", r"+\frac{1}{2}", ["+1", "/2"]),
    ("-1/2", r"-\frac{1}{2}", ["-1/2"]),
    ("x-1/2", r"x-\frac{1}{2}", ["x", "-1/2"]),
    ("x+1/2", r"x+\frac{1}{2}", ["x", "+", "1/2"]),
    ("a^-2", "a^{-2}", ["a", "^-2"]),
    ("x²", "x^{2}", ["x", "²"]),
    ("x^2+y^2", "x^{2}+y^{2}", ["x^2", "+", "y^2"]),
    ("25%", r"25\%", ["25", "%"]),
    ("10%", r"10\%", ["10", "%"]),
    ("1/2", r"\frac{1}{2}", ["1", "/", "2"]),
    ("3x-2", "3x-2", ["3x", "-2"]),
    ("a=2", "a=2", ["a", "=", "2"]),
    ("√2", r"\sqrt{2}", ["√", "2"]),
    ("3.5", "3.5", ["3.5"]),
    ("-x", "-x", ["-", "x"]),
    ("2x", "2x", ["2", "x"]),
    ("3+4", "3+4", ["3", "+", "4"]),
]


@pytest.mark.parametrize("form, latex, split", FORMS, ids=[f[0] for f in FORMS])
def test_expansion_equals_zh_math_reading_on_the_equivalent_latex(form, latex, split):
    """听写的写法展开后，与 zh-math-reading 读 `$等价的 LaTeX$` 的结果逐字相同（空格不计）：连着写的词，和拆成几个词的一样。"""
    expected = _strip(normalize(f"${latex}$"))
    assert _strip(heard.expand(form)) == expected
    words = [{"word": tok, "start": i * 0.2, "end": (i + 1) * 0.2} for i, tok in enumerate(split)]
    assert _strip(heard.expand_words(words)) == expected


def test_lone_operator_between_operands_is_binary():
    """单独的 “+” 两边是别的词：读作“加”（和 “a+b” 里一样），不读作 zh-math-reading 对单独 “+” 的“正”。"""
    words = [{"word": "三", "start": 0, "end": 1}, {"word": "+", "start": 1, "end": 2}, {"word": "四", "start": 2, "end": 3}]
    assert _strip(heard.expand_words(words)) == "三加四" == _strip(normalize("$3+4$"))


def _timed(tokens, d):
    """每个词 d 秒，依次排开（标点也是 d 秒）。"""
    return [{"word": tok, "start": round(i * d, 6), "end": round((i + 1) * d, 6)} for i, tok in enumerate(tokens)]


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("script_text, expected_index", [
    ("{@a}$10$，10，10。", 0),     # 稿子：十，10，10 —— a 在第一个数（读作十）前面
    ("$10$，{@a}10，10。", 2),      # 稿子：十，10，10 —— a 在第二个数前面
    ("$10$，10，{@a}10。", 4),      # 稿子：十，10，10 —— a 在第三个数前面
    ("10，{@a}$10$，10。", 2),      # 稿子：10，十，10 —— 公式里的 10 读作“十”，在第二个数前面
    ("10，10，{@a}10。", 4),        # 稿子全是裸数字：都保留数字
])
def test_repeated_numbers_pick_their_reading_jointly_and_keep_bookmark_time(script_text, expected_index, d, no_audio):
    """复现 1：同一句里多个 10：每个数字的读法由整句的对齐决定，书签时刻 = 它前面的词的时长之和。"""
    tokens = ["10", "，", "10", "，", "10", "。"]
    words = _timed(tokens, d)
    res = _run(script_text, tokens, words=words)
    assert res.how == {"a": "matched"}, (script_text, res.how)
    assert res.marks["a"] == pytest.approx(expected_index * d, abs=0.002)
    assert res.ratio == 1.0


@pytest.mark.parametrize("x_end", [0.1, 0.4, 0.9])
def test_power_keeps_the_base_letter_at_its_own_span(x_end, no_audio):
    """复现 2：x [0, x_end]、² [x_end, 1.0]：x 的时间段保持不变（不论两词的时长比例），“的平方”放在 ² 的时间段里。"""
    words = [{"word": "x", "start": 0.0, "end": x_end}, {"word": "²", "start": x_end, "end": 1.0},
             {"word": "的值", "start": 1.0, "end": 1.5}]
    chars = heard.chars_from_words(words)
    assert [c.ch for c in chars[:4]] == ["x", "的", "平", "方"]
    assert (chars[0].start, chars[0].end) == (pytest.approx(0.0), pytest.approx(x_end))
    assert chars[1].start == pytest.approx(x_end) and chars[3].end == pytest.approx(1.0)
    res = _run("{@a}$x^2$的值。", ["x", "²", "的值"], words=words)
    assert res.marks["a"] == pytest.approx(0.0, abs=0.002)  # 句首书签仍是 x 的起点，不晚


@pytest.mark.parametrize("split_at", [0.1, 0.5, 0.9])
def test_fraction_split_or_joined_gets_the_same_span(split_at):
    """复现 2（分数）：“1/2” 无论连着写还是拆成三个词，四个读法字都平分在整个表达式的时间段 [0, 1.0] 里。"""
    joined = heard.chars_from_words([{"word": "1/2", "start": 0.0, "end": 1.0}])
    split = heard.chars_from_words([{"word": "1", "start": 0.0, "end": split_at},
                                    {"word": "/", "start": split_at, "end": split_at},
                                    {"word": "2", "start": split_at, "end": 1.0}])
    for chars in (joined, split):
        assert [c.ch for c in chars] == ["二", "分", "之", "一"]
        assert [round(c.start, 6) for c in chars] == [0.0, 0.25, 0.5, 0.75]
        assert [round(c.end, 6) for c in chars] == [0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("d", [0.5, 1.0, 2.0])
def test_sign_word_reads_like_the_script_and_keeps_bookmark_time(d, no_audio):
    """复现 3：“{@a}$+3$”：听写的 “+3” 读作“正三”（和稿子一样），书签在句首（0 秒）。"""
    res = _run("{@a}$+3$，好。", ["+3", "，好。"], words=_timed(["+3", "，好。"], d))
    assert res.how == {"a": "matched"} and res.marks["a"] == pytest.approx(0.0, abs=0.002)


@pytest.mark.parametrize("split_at", [0.2, 1.0, 1.8])
def test_percent_split_across_words_spreads_over_the_whole_span(split_at):
    """复现 2（百分数）：“25” [0, split] + “%” [split, 2.0]：六个读法字平分 [0, 2.0]，与两词的时长比例无关（不把数字的时间段压缩）。"""
    chars = heard.chars_from_words([{"word": "25", "start": 0.0, "end": split_at},
                                    {"word": "%", "start": split_at, "end": 2.0}])
    assert "".join(c.ch for c in chars) == "百分之二十五"
    assert [round(c.start, 6) for c in chars] == pytest.approx([i * 2 / 6 for i in range(6)])
    assert chars[-1].end == pytest.approx(2.0)
