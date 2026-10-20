"""对齐警告和审片问题追溯回场景文件的行（beats[i].say[j] (line L, t=...)）。不联网：审片用写好的模型输出。"""

import json
import shutil
from pathlib import Path

from scene_spec.loader import load_file

from voice import align, review, script, trace

FIXTURES = Path(__file__).parent / "fixtures" / "trace"
SCENE = FIXTURES / "episode.scene.yaml"
TIMINGS = {
    "Hook": {"cues": [{"start": 0.3, "end": 2.0}, {"start": 2.45, "end": 4.0}], "total": 4.0},
    "Think": {"cues": [{"start": 0.3, "end": 1.5}], "total": 1.5},
    "Solve": {"cues": [{"start": 0.3, "end": 1.8}, {"start": 2.25, "end": 4.0}], "total": 4.0},
}


def line_of(snippet: str) -> int:
    """场景文件里含 snippet 的那一行（1 起）。"""
    lines = SCENE.read_text(encoding="utf-8").splitlines()
    return next(i for i, text in enumerate(lines, start=1) if snippet in text)


def audio_dir(tmp_path: Path) -> Path:
    audio = tmp_path / "audio"
    audio.mkdir()
    for beat_id, timing in TIMINGS.items():
        (audio / f"{beat_id}.timing.json").write_text(json.dumps(timing), encoding="utf-8")
    return audio


def test_scene_refs_point_at_the_text_lines():
    refs = trace.scene_refs(load_file(SCENE), SCENE)
    assert list(refs) == ["Hook", "Think", "Show", "Solve"]
    assert refs["Hook"].line == line_of("id: Hook") and refs["Show"].says == []
    hook = refs["Hook"].says
    assert [s.key for s in hook] == ["h1", "h2"]
    assert hook[0].line == line_of("先看这道题") and hook[1].line == line_of("你猜答案")
    solve = refs["Solve"].says
    assert [s.key for s in solve] == ["0", "s2"] and solve[1].line == line_of("再把差加回去")
    assert trace.describe(solve[1], 12.3, 15.8) == f"beats[3].say[1] (line {line_of('再把差加回去')}, t=12.300-15.800)"


def test_english_refs_point_at_en_lines():
    text = "beats:\n  - id: B\n    say:\n      - text: 中文。\n        pace: normal\n        en: English.\n"
    doc = {"beats": [{"id": "B", "say": [{"text": "English.", "pace": "normal"}]}]}
    assert trace.scene_refs(doc, text, "en")["B"].says[0].line == 6
    assert trace.scene_refs(doc, text, "zh")["B"].says[0].line == 4


def test_align_warnings_carry_scene_path_and_line():
    doc = load_file(SCENE)
    refs = trace.scene_refs(doc, SCENE)
    beat = script.beats(doc)[3]
    ok = align.SentenceResult("0", 1.0, marks={"a": 0.5}, how={"a": "matched"}, index=0, start=0.3, end=1.8)
    bad = align.SentenceResult("s2", 0.6, marks={"b": 3.1, "c": 3.5}, how={"b": "interpolated", "c": "estimated"},
                               missing=["加回"], heard="再把差去", confirmed=["加回"], index=1, start=2.25, end=4.0)
    out = align.warnings([ok, bad], refs, beat.id)
    where = f"beats[3].say[1] (line {line_of('再把差加回去')}, t=2.250-4.000)"
    assert len(out) == 3 and all(line.startswith(where + ":") for line in out)
    assert "{@b}" in out[0] and "interpolated" in out[0] and "{@c}" in out[1] and "estimated" in out[1]
    assert "没听到 ['加回']" in out[2]
    assert align.warnings([bad], None, "Solve")[0].startswith("Solve[s2] (t=2.250-4.000)")


def test_timeline_from_chapters_handles_pause_no_audio_and_freezes(tmp_path):
    doc = load_file(SCENE)
    refs = trace.scene_refs(doc, SCENE)
    timeline = trace.build_timeline(refs, audio_dir(tmp_path), FIXTURES / "episode.chapters.json", doc)
    assert timeline.source == "chapters"
    spans = {(s.ref.beat_id, s.ref.key): (round(s.start, 3), round(s.end, 3)) for s in timeline.sentences}
    assert spans[("Think", "0")] == (4.8, 6.0)
    assert spans[("Solve", "0")] == (11.3, 13.3)  # 1.0 秒处定格 0.5 秒：句中被拉长
    assert spans[("Solve", "s2")] == (13.75, 15.5)  # 定格之后的句子整体后移
    beat, say = timeline.locate(7.0, 8.0)  # 想一想的暂停：只对到 beat
    assert beat.ref.id == "Think" and say is None
    assert timeline.locate(20, 21) == (None, None)


def test_estimated_timeline_without_chapters(tmp_path):
    doc = {"beats": [{"id": "Hook", "say": [{"pace": "normal", "text": "一。"}, {"pace": "normal", "text": "二。"}]},
                     {"id": "Think", "pause": {"hold": 3}, "say": [{"pace": "normal", "text": "三。"}]}]}
    refs = trace.scene_refs(doc)
    audio = audio_dir(tmp_path)
    timeline = trace.build_timeline(refs, audio, None, doc)
    assert timeline.source == "estimated"
    assert [(round(b.start, 2), round(b.end, 2)) for b in timeline.beats] == [(0.0, 4.28), (4.28, 9.06)]
    (audio / "Think.timing.json").unlink()
    assert trace.build_timeline(refs, audio, None, doc) is None  # 缺一个 beat 的时长就不估


def issue(category, start, end, **extra):
    return {"category": category, "severity": "minor", "start": start, "end": end,
            "evidence": f"{category} 在 {start}", "suggested_fix": "改", **extra}


def test_review_fills_beat_sentence_and_line_from_timeline(tmp_path):
    work = tmp_path / "out"
    work.mkdir()
    video = work / "episode.mp4"
    video.write_bytes(b"not a real video")
    shutil.copy(FIXTURES / "episode.chapters.json", work / "episode.chapters.json")  # kit 写在成片旁边
    issues = [
        issue("layout_overlap", 1.0, 1.5),                     # Hook 第一句
        issue("tts_filler", 2.6, 3.0),                         # Hook 第二句
        issue("av_sync", 2.05, 2.35),                          # 两句之间 → 只对到 beat
        issue("pedagogy", 7.0, 8.0),                           # 想一想的暂停
        issue("legibility", 10.0, 10.5),                       # 没有旁白的 beat
        issue("math", 15.2, 15.4),                             # 定格之后的句子
        issue("other", 1.2, 1.2),                              # 一个时间点
        issue("seam", 11.5, 12.0, beat="Solve", sentence="0"),  # 模型填对了
        issue("tts_pronunciation", 14.0, 14.2, beat="Hook"),   # 模型填的和时间对不上
        issue("out_of_frame", 16.5, 16.9),                     # 片尾，不在任何 beat 里
    ]
    reviewer_text = json.dumps({"summary": "看完了。", "issues": issues}, ensure_ascii=False)
    report = review.review_episode(
        video, reviewer=lambda **_: review.Turn(text=reviewer_text, calls=[(10, 10)]), out_json=work / "episode.review.json",
        model="gemini-4-argon", spec=SCENE, audio=audio_dir(tmp_path), probe=lambda _: 17.0, cut=None,
        log=lambda *_: None,
    )
    got = {item["category"]: item for item in report["issues"]}
    assert report["sources"]["timeline"] == "chapters"

    def where(item):
        return item.get("beat"), item.get("sentence"), item.get("line")

    hook1, hook2, solve2 = line_of("先看这道题"), line_of("你猜答案"), line_of("再把差加回去")
    # 落在旁白句子时间窗里的问题：beat、sentence、行号全部补上
    assert where(got["layout_overlap"]) == ("Hook", "h1", hook1)
    assert got["layout_overlap"]["path"] == f"beats[0].say[0] (line {hook1}, t=1.000-1.500)"
    assert where(got["tts_filler"]) == ("Hook", "h2", hook2)
    assert where(got["math"]) == ("Solve", "s2", solve2)
    assert where(got["other"]) == ("Hook", "h1", hook1)
    # 句间、暂停、没有旁白的 beat：只到 beat
    assert where(got["av_sync"]) == ("Hook", None, line_of("id: Hook"))
    assert got["av_sync"]["path"].startswith("beats[0] (line")
    assert where(got["pedagogy"]) == ("Think", None, line_of("id: Think"))
    assert where(got["legibility"]) == ("Show", None, line_of("id: Show"))
    # 模型填的：对得上就补行号；对不上保留模型的值，写明原因
    assert where(got["seam"]) == ("Solve", "0", line_of("先算差")) and "trace_note" not in got["seam"]
    mismatch = got["tts_pronunciation"]
    assert where(mismatch) == ("Hook", None, line_of("id: Hook"))
    assert "beats[3].say[1]" in mismatch["trace_note"] and "保留模型的" in mismatch["trace_note"]
    # 片尾不在任何 beat：不动
    assert {"beat", "sentence", "line", "path"}.isdisjoint(got["out_of_frame"])

    gate = {item["message"].split(":")[0].split()[-1]: item["path"] for item in report["gate"]["issues"]}
    assert gate["layout_overlap"] == got["layout_overlap"]["path"]
    assert gate["out_of_frame"] == "t=16.500-16.900"
    review.validate_report(report)
    schema = json.loads((review.prompt_dir("v1") / "report.schema.json").read_text(encoding="utf-8"))
    allowed = set(schema["$defs"]["reportIssue"]["properties"])
    assert all(set(item) <= allowed for item in report["issues"])
    md = (work / "episode.review.md").read_text(encoding="utf-8")
    assert f"`beats[3].say[1]` 第 {solve2} 行" in md and "保留模型的" in md


def test_review_without_spec_or_timeline_keeps_issues_as_before(tmp_path):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"x")
    text = json.dumps({"summary": "好。", "issues": [issue("math", 1.0, 2.0)]}, ensure_ascii=False)
    report = review.review_episode(video, reviewer=lambda **_: review.Turn(text=text, calls=[(1, 1)]),
                                   out_json=tmp_path / "r.json", model="gemini-4-argon", spec=SCENE,
                                   probe=lambda _: 5.0, log=lambda *_: None)
    assert {"beat", "sentence", "line", "path"}.isdisjoint(report["issues"][0])
    assert report["gate"]["issues"][0]["path"] == "t=1.000-2.000" and report["sources"]["timeline"] is None


def test_freeze_on_a_non_integer_boundary_is_not_lost_to_float_error(tmp_path):
    """复现：start=11.0、定格 time=12.3、hold=0.5，还原出 at=1.3000000000000007；cue 正好在 1.3 时必须加上 hold。"""
    chapter = {"id": "B", "start": 11.0, "end": 16.0, "steps": [{"time": 12.3, "hold": 0.5}]}
    assert trace._freezes(chapter) == [(1300, 500)]
    assert trace._to_video(1.3, trace._freezes(chapter)) == 1.8
    assert trace._to_video(1.299, trace._freezes(chapter)) == 1.299
    doc = {"beats": [{"id": "B", "say": [{"pace": "normal", "text": "一。"}, {"pace": "normal", "text": "二。"}]}]}
    refs = trace.scene_refs(doc)
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "B.timing.json").write_text(json.dumps({"cues": [{"start": 0.3, "end": 0.85}, {"start": 1.3, "end": 3.0}],
                                                     "total": 3.0}))
    chapters = tmp_path / "e.chapters.json"
    chapters.write_text(json.dumps({"chapters": [chapter]}))
    timeline = trace.build_timeline(refs, audio, chapters, doc)
    second = timeline.sentences[1]
    assert (second.start, second.end) == (12.8, 14.5)  # 不是 (12.3, 14.5)：定格的静音不能算进这句
    assert timeline.locate(12.4, 12.6)[1] is None  # 定格里的问题只对到 beat


def test_model_sentence_in_a_silence_window_gets_a_trace_note(tmp_path):
    """复现：模型填了对的 beat 和某一句，但时间落在这个 beat 的句间静音 / 暂停里：保留模型的值，并写明冲突。"""
    doc = load_file(SCENE)
    refs = trace.scene_refs(doc, SCENE)
    timeline = trace.build_timeline(refs, audio_dir(tmp_path), FIXTURES / "episode.chapters.json", doc)
    issues = [
        {"start": 2.05, "end": 2.35, "beat": "Hook", "sentence": "h2"},  # Hook 两句之间的静音
        {"start": 7.0, "end": 8.0, "beat": "Think", "sentence": "0"},    # 想一想的暂停
        {"start": 2.05, "end": 2.35, "sentence": "h1"},                  # 只给了句子
        {"start": 2.6, "end": 3.0, "beat": "Hook", "sentence": "h2"},    # 对得上：不写说明
        {"start": 10.0, "end": 10.5, "beat": "Show"},                    # 没有旁白的 beat：本来就只到 beat
    ]
    review.trace_issues(issues, refs, timeline)
    gap, pause, only_sentence, fine, silent = issues
    assert (gap["beat"], gap["sentence"], gap["line"]) == ("Hook", "h2", line_of("你猜答案"))
    assert "句间静音或暂停" in gap["trace_note"] and "beats[0].say[1]" in gap["trace_note"]
    assert "t=2.450-4.000" in gap["trace_note"] and "保留模型的" in gap["trace_note"]
    assert (pause["sentence"], pause["line"]) == ("0", line_of("停下来想一想")) and "句间静音或暂停" in pause["trace_note"]
    assert only_sentence["beat"] == "Hook" and only_sentence["sentence"] == "h1"
    assert "句间静音或暂停" in only_sentence["trace_note"]
    assert "trace_note" not in fine and "trace_note" not in silent
