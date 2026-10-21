"""审片不联网：用录好的模型输出核对分段、schema、阈值、报告和费用。"""

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from voice import review
from voice.review_gemini import (
    GeminiReviewer,
    call_with_retries,
    client_mode,
    generate_text,
    is_transient,
    tokens_from_usage,
    wait_until_active,
)

FIXTURES = Path(__file__).parent / "fixtures" / "review"
WHEN = date(2026, 10, 20)


class Scripted:
    def __init__(self, texts, calls=None):
        self.texts = list(texts)
        self.calls = calls or [(1000, 100)]
        self.seen = []

    def __call__(self, *, video, system, user, chunk):
        self.seen.append({"video": video, "system": system, "user": user, "chunk": chunk})
        return review.Turn(text=self.texts.pop(0), calls=list(self.calls))


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "episode.mp4"
    path.write_bytes(b"not a real video")
    return path


def _run(tmp_path, texts, *, duration=30.0, model="gemini-4-argon", thresholds=None, spec=None, audio=None,
         calls=None, max_chunk=300.0, overlap=8.0):
    video = _video(tmp_path)
    scripted = Scripted(texts, calls)
    cuts = []

    def cut(src, start, end, dest):
        cuts.append((start, end))
        dest.write_bytes(b"clip")

    report = review.review_episode(
        video, reviewer=scripted, out_json=tmp_path / "out" / "episode.review.json", model=model,
        spec=spec, audio=audio, thresholds=thresholds or review.load_thresholds(None),
        max_chunk_seconds=max_chunk, overlap_seconds=overlap, probe=lambda path: duration, cut=cut,
        today=WHEN, log=lambda *_: None,
    )
    return report, scripted, cuts


def test_plan_chunks_short_and_overlapping():
    assert review.plan_chunks(30, 300, 8) == [review.Chunk(0, 0.0, 30.0)]
    assert review.plan_chunks(300, 300, 8) == [review.Chunk(0, 0.0, 300.0)]
    chunks = review.plan_chunks(400, 300, 8)
    assert [(c.start, c.end) for c in chunks] == [(0.0, 300.0), (292.0, 400.0)]
    with pytest.raises(review.ReviewError):
        review.plan_chunks(0, 300, 8)
    with pytest.raises(review.ReviewError):
        review.plan_chunks(10, 300, 300)


def test_parse_recorded_response_and_reject_bad_schema():
    parsed = review.parse_model_output(FIXTURES.joinpath("chunk0.json").read_text(encoding="utf-8"), chunk_len=300)
    assert parsed["issues"][0]["category"] == "layout_overlap"
    assert parsed["issues"][0]["start"] == 12.4
    fenced = "```json\n" + json.dumps({"summary": "好的。", "issues": []}) + "\n```"
    assert review.parse_model_output(fenced, chunk_len=10)["issues"] == []
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output(FIXTURES.joinpath("invalid.json").read_text(encoding="utf-8"), chunk_len=30)
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output('{"summary": "x", "issues": [{"category": "math", "severity": "major", '
                                   '"start": 3, "end": 1, "evidence": "a", "suggested_fix": "b"}]}', chunk_len=10)
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output('{"summary": "x", "issues": [{"category": "math", "severity": "major", '
                                   '"start": 20, "end": 21, "evidence": "a", "suggested_fix": "b"}]}', chunk_len=5)
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output('{"summary": "  ", "issues": []}', chunk_len=5)
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output('{"summary": "x", "issues": [{"category": "math", "severity": "major", '
                                   '"start": true, "end": 1, "evidence": "a", "suggested_fix": "b"}]}', chunk_len=5)


def test_dedupe_merges_overlap_and_keeps_worse_severity():
    # 这两句是录好的相邻窗口对同一处音画错位的转述，相似度高于门槛，应当并成一条。
    merged = review.dedupe_issues([
        {"category": "av_sync", "severity": "major", "start": 292.2, "end": 296.0,
         "evidence": "旁白已经在说二阶差数列，高亮还停在一阶差那一行。",
         "suggested_fix": "提前", "chunk": 0},
        {"category": "av_sync", "severity": "blocker", "start": 292.4, "end": 296.2,
         "evidence": "这段开头旁白已经在说二阶差，画面还停在一阶。",
         "suggested_fix": "重做", "chunk": 1},
        {"category": "math", "severity": "minor", "start": 292.5, "end": 293.0, "evidence": "数字",
         "suggested_fix": "改", "chunk": 1},
    ])
    assert len(merged) == 2
    av = next(item for item in merged if item["category"] == "av_sync")
    assert av["severity"] == "blocker" and av["evidence"] == "这段开头旁白已经在说二阶差，画面还停在一阶。"
    assert av["start"] == 292.2 and av["end"] == 296.2 and av["chunk"] == 1


def test_thresholds_pass_fail_and_gate_shape():
    major = {"category": "layout_overlap", "severity": "major", "code": "RV100", "start": 1, "end": 2,
             "evidence": "叠了", "suggested_fix": "挪开", "id": "i1", "chunk": 0}
    minor = {"category": "math", "severity": "minor", "code": "RV310", "start": 3, "end": 4,
             "evidence": "差 1", "suggested_fix": "改成 121", "id": "i2", "chunk": 0}
    defaults = review.load_thresholds(None)
    passed, violations = review.decide([minor], defaults)
    assert passed and not violations
    passed, violations = review.decide([major], defaults)
    assert not passed and not violations
    strict = {"fail_severities": ["blocker"], "max_counts": {}, "fail_categories": {"math": "minor"}}
    assert review.decide([minor], strict) == (False, [])
    capped = {"fail_severities": [], "max_counts": {"minor": 0}, "fail_categories": {}}
    passed, violations = review.decide([minor], capped)
    assert not passed and "minor" in violations[0]
    gate = review.project_gate([major, minor], defaults, [])
    assert gate["ok"] is False
    assert [item["level"] for item in gate["issues"]] == ["error", "warning"]
    assert gate["issues"][0]["code"] == "RV100" and gate["issues"][0]["path"] == "t=1.000-2.000"
    assert set(gate["issues"][0]) == {"level", "code", "path", "message"}


def test_threshold_file_rejects_unknown_fields(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"fail_categories": {"math": "minor"}, "max_counts": {"info": 3}}), encoding="utf-8")
    loaded = review.load_thresholds(path)
    assert loaded["fail_categories"] == {"math": "minor"} and loaded["fail_severities"] == ["blocker", "major"]
    path.write_text('{"oops": true}', encoding="utf-8")
    with pytest.raises(review.ReviewError):
        review.load_thresholds(path)


def test_price_follows_published_schedule():
    assert review.price_call("gemini-4-argon", 1_000_000, 0, WHEN) == (2.0, True)
    assert review.price_call("gemini-4-argon", 0, 1_000_000, date(2026, 12, 1)) == (20.0, True)
    assert review.price_call("gemini-3.1-pro-preview", 200_001, 0, WHEN)[0] == pytest.approx(0.800004)
    assert review.price_call("gemini-3.8-flash", 1_000_000, 0, WHEN) == (0.75, True)
    assert review.price_call("gemini-3.8-flash", 1_000_000, 0, date(2027, 1, 1)) == (1.5, True)
    usd, known = review.price_call("some-other-model", 1_000_000, 0, WHEN)
    assert usd == 2.0 and known is False
    assert review.estimate_tokens("abcd", 10, "high") == (3006, 800)
    assert review.estimate_tokens("", 10, "low") == (1000, 800)


def test_prompt_files_list_every_category():
    system, user, schema = review.load_prompt("v1")
    for category in review.CATEGORIES:
        assert category in system
    assert set(schema["$defs"]["issue"]["properties"]["category"]["enum"]) == set(review.CATEGORIES)
    assert "{{CONTEXT}}" in user and "{{SCHEMA}}" in user
    report_schema = json.loads((review.prompt_dir("v1") / "report.schema.json").read_text(encoding="utf-8"))
    assert report_schema["properties"]["gate"]["$ref"] == "#/$defs/gate"
    assert "scene_spec.report.Report.to_dict" in report_schema["description"]
    assert "check_timing" in report_schema["description"] and "{beats: ...}" in report_schema["description"]
    readme = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert "check_timing" in readme and "{beats: ...}" in readme and "run_id" in readme
    with pytest.raises(review.ReviewError):
        review.load_prompt("v9")


def test_recorded_chunks_merge_offsets_cost_and_markdown(tmp_path):
    texts = [(FIXTURES / name).read_text(encoding="utf-8") for name in ("chunk0.json", "chunk1.json")]
    report, scripted, cuts = _run(tmp_path, texts, duration=400)
    assert [item["chunk"].index for item in scripted.seen] == [0, 1]
    assert cuts == [(0.0, 300.0), (292.0, 400.0)]
    assert "从这一段的第一帧" in scripted.seen[0]["user"]
    assert "layout_overlap" in scripted.seen[0]["system"]
    by_cat = {item["category"]: item for item in report["issues"]}
    assert set(by_cat) == {"layout_overlap", "av_sync", "math", "tts_filler"}
    assert by_cat["layout_overlap"]["code"] == "RV100"
    assert by_cat["layout_overlap"]["start"] == 12.4
    assert by_cat["av_sync"]["start"] == 292.2 and by_cat["av_sync"]["end"] == 296.2
    assert by_cat["math"]["code"] == "RV310" and by_cat["math"]["start"] == 312
    assert by_cat["tts_filler"]["start"] == 322
    assert report["passed"] is False and report["ok"] is False
    assert report["schema"] == "animind.review/v1"
    assert [item["level"] for item in report["gate"]["issues"]] == ["error", "error", "error", "warning"]
    assert report["usage"]["calls"] == 2
    assert report["usage"]["input_tokens"] == 2000 and report["usage"]["output_tokens"] == 200
    assert report["usage"]["est_usd"] == pytest.approx(0.006)
    md = (tmp_path / "out" / "episode.review.md").read_text(encoding="utf-8")
    assert "未通过" in md and "RV310" in md and "00:12.4–00:15.8" in md
    assert "调用 2 次，约 $0.0060" in md
    lines = (tmp_path / "out" / "review_usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["model"] == "gemini-4-argon" and row["price_known"] is True and row["prompt_version"] == "v1"
    review.validate_report(report)


def test_schema_retry_then_pass_logs_both_calls(tmp_path):
    bad = FIXTURES.joinpath("invalid.json").read_text(encoding="utf-8")
    good = json.dumps({"summary": "看完了，没有要退回的问题。", "issues": []}, ensure_ascii=False)
    report, scripted, cuts = _run(tmp_path, [bad, good], duration=20)
    assert cuts == []  # 短于一段，不重新编码
    assert len(scripted.seen) == 2
    assert "上一次的输出不能用" in scripted.seen[1]["user"]
    assert report["passed"] is True and report["issues"] == []
    assert report["usage"]["calls"] == 2
    md = (tmp_path / "out" / "episode.review.md").read_text(encoding="utf-8")
    assert "通过" in md


def test_minor_only_passes_and_secrets_are_redacted(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sekret-key-xyz")
    text = json.dumps({
        "summary": "摘要里出现了 sekret-key-xyz。",
        "issues": [{
            "category": "tts_filler", "severity": "minor", "start": 1, "end": 1.5,
            "evidence": "多了嗯", "suggested_fix": "重念",
        }],
    }, ensure_ascii=False)
    report, _, _ = _run(tmp_path, [text], duration=12)
    assert report["passed"] is True
    assert report["gate"]["issues"][0]["level"] == "warning"
    assert "sekret-key-xyz" not in json.dumps(report, ensure_ascii=False)
    blob = (tmp_path / "out" / "episode.review.md").read_text(encoding="utf-8")
    blob += (tmp_path / "out" / "review_usage.jsonl").read_text(encoding="utf-8")
    assert "sekret-key-xyz" not in blob and "***" in report["summary"]


def test_context_includes_script_and_listenback(tmp_path):
    spec = tmp_path / "demo.scene.yaml"
    spec.write_text(
        "spec: '0.1'\nmeta:\n  id: demo\n  title: 演示\npedagogy:\n  insight: 先做差再加回去\n"
        "beats:\n  - id: Opening\n    say:\n      - {id: s1, pace: normal, text: '别慌。{@a}先看前五项。'}\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / "Opening.timing.json").write_text(json.dumps({
        "cues": [{"start": 0.3, "end": 2.0}], "total": 2.2,
        "marks": {"s1": {"a": 1.1}},
        "check": {"s1": {"ratio": 0.8, "missing": ["别慌"], "extra": ["嗯"], "heard": "嗯。先看前五项。"}},
    }, ensure_ascii=False), encoding="utf-8")
    text, episode_id, title = review.build_context(spec, audio, "zh")
    assert episode_id == "demo" and title == "演示"
    assert "别慌。先看前五项。" in text
    assert "没听到" in text and "a@1.1" in text
    assert "beat 内部" in text and "先做差再加回去" in text
    bare, _, _ = review.build_context(None, None, "zh")
    assert "没有场景文件" in bare


def test_normalize_model_strips_openrouter_prefix():
    assert review.normalize_model("google/gemini-4-argon") == "gemini-4-argon"
    assert review.normalize_model("gemini-3.8-flash") == "gemini-3.8-flash"


def test_cli_writes_report_and_exits_nonzero_on_fail(tmp_path, monkeypatch):
    from voice.cli import main

    video = _video(tmp_path)
    scripted = Scripted([json.dumps({
        "summary": "标题压住了题卡。",
        "issues": [{"category": "layout_overlap", "severity": "major", "start": 2, "end": 4,
                    "evidence": "两行字叠在一起。", "suggested_fix": "把题卡下移。"}],
    }, ensure_ascii=False)])
    monkeypatch.setattr(review, "probe_duration", lambda path: 40.0)
    monkeypatch.setattr(review, "default_reviewer", lambda *args, **kwargs: scripted)
    code = main(["review", str(video), "--out", str(tmp_path / "report.json"), "--model", "gemini-3.8-flash"])
    assert code == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["model"] == "gemini-3.8-flash" and report["passed"] is False
    assert (tmp_path / "report.md").exists()
    assert main(["review", str(tmp_path / "missing.mp4")]) == 2


def test_retries_usage_and_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "ANIMIND_REVIEW_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(review.ReviewError):
        client_mode()
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert client_mode() == "gemini"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    assert client_mode() == "vertex"
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert client_mode() == "gemini"
    monkeypatch.setenv("ANIMIND_REVIEW_BACKEND", "vertex")
    assert client_mode() == "vertex"
    monkeypatch.setenv("ANIMIND_REVIEW_BACKEND", "nope")
    with pytest.raises(review.ReviewError):
        client_mode()

    sleeps = []

    def flaky():
        flaky.n += 1
        if flaky.n == 1:
            raise RuntimeError("503 temporarily unavailable")
        return "ok"

    flaky.n = 0
    assert call_with_retries(flaky, attempts=3, sleep=sleeps.append) == "ok"
    assert sleeps == [1.0]

    class Denied(Exception):
        code = 400

    with pytest.raises(Denied):
        call_with_retries(lambda: (_ for _ in ()).throw(Denied()), attempts=3, sleep=sleeps.append)
    assert is_transient(RuntimeError("timeout")) and not is_transient(Denied())
    usage = SimpleNamespace(prompt_token_count=10, tool_use_prompt_token_count=2,
                            candidates_token_count=4, thoughts_token_count=1)
    assert tokens_from_usage(usage) == (12, 5) and tokens_from_usage(None) == (0, 0)

    class Models:
        def generate_content_stream(self, **kwargs):
            assert kwargs["model"] == "gemini-4-argon"
            return [SimpleNamespace(text='{"a":', usage_metadata=None), SimpleNamespace(text="1}", usage_metadata=usage)]

    text, meta = generate_text(SimpleNamespace(models=Models()), "gemini-4-argon", [], None)
    assert text == '{"a":1}' and tokens_from_usage(meta) == (12, 5)

    class Files:
        def __init__(self):
            self.state = "PROCESSING"

        def get(self, name):
            self.state = "ACTIVE"
            return SimpleNamespace(name=name, state=SimpleNamespace(name="ACTIVE"), uri="u")

    files = Files()
    got = wait_until_active(SimpleNamespace(files=files), SimpleNamespace(name="files/1", state=SimpleNamespace(name="PROCESSING")),
                            sleep=lambda seconds: None, attempts=3)
    assert got.state.name == "ACTIVE"


def test_sdk_parts_keep_resolution_and_processing():
    pytest.importorskip("google.genai")
    from google.genai import types

    from voice.review_gemini import _file_part, _inline_part

    resolution = {"level": "media_resolution_high"}
    inline = _inline_part(types, b"xyz", "video/mp4", resolution, "STATIC")
    uploaded = _file_part(types, "https://example.com/a.mp4", "video/mp4", resolution, "AGENTIC")
    assert inline.inline_data.data == b"xyz"
    assert "HIGH" in str(inline.media_resolution) and "STATIC" in str(inline.media_processing)
    assert uploaded.file_data.file_uri.endswith("a.mp4")
    assert "HIGH" in str(uploaded.media_resolution) and "AGENTIC" in str(uploaded.media_processing)


def test_ffmpeg_probe_and_cut(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("需要 ffmpeg")
    src = tmp_path / "clip.mp4"
    made = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)],
        capture_output=True, text=True,
    )
    if made.returncode != 0:
        pytest.skip(made.stderr)
    duration = review.probe_duration(src)
    assert 1.5 < duration < 2.5
    dest = tmp_path / "part.mp4"
    review.cut_clip(src, 0.4, 1.2, dest)
    piece = review.probe_duration(dest)
    assert 0.6 < piece < 1.2


def _model_json(**overrides) -> str:
    issue = {
        "category": "math", "severity": "major", "start": 1, "end": 2,
        "evidence": "算错了。", "suggested_fix": "改掉。",
    }
    issue.update(overrides)
    return json.dumps({"summary": "看过了。", "issues": [issue]}, ensure_ascii=False)


@pytest.mark.parametrize("overrides", [{"category": []}, {"severity": {}}])
def test_unhashable_category_or_severity_is_schema_error(overrides):
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output(_model_json(**overrides), chunk_len=10)


def test_unhashable_category_retries_instead_of_raising_typeerror(tmp_path):
    good = json.dumps({"summary": "看完了，没有要退回的问题。", "issues": []}, ensure_ascii=False)
    report, scripted, _ = _run(tmp_path, [_model_json(category=[]), good], duration=20)
    assert len(scripted.seen) == 2
    assert "category" in scripted.seen[1]["user"]
    assert report["passed"] is True and report["issues"] == []


@pytest.mark.parametrize("payload", [
    {"fail_severities": [[]]},
    {"fail_severities": [{}]},
    {"fail_severities": "major"},
    {"fail_categories": {"math": []}},
    {"fail_categories": {"math": {}}},
    {"max_counts": {"minor": []}},
])
def test_threshold_file_typechecks_before_membership(tmp_path, payload):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(review.ReviewError):
        review.load_thresholds(path)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_nonfinite_timestamps_are_schema_errors(token):
    text = (
        '{"summary": "看过了。", "issues": [{"category": "math", "severity": "major", "start": '
        + token
        + ', "end": 1, "evidence": "算错了。", "suggested_fix": "改掉。"}]}'
    )
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output(text, chunk_len=10)
    with pytest.raises(review.ReviewSchemaError):
        review._as_float(float(token), "start")


def test_dedupe_keeps_same_segment_and_dissimilar_math_errors():
    same_window = review.dedupe_issues([
        {"category": "math", "severity": "minor", "start": 10, "end": 12,
         "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121", "chunk": 0},
        {"category": "math", "severity": "minor", "start": 10.2, "end": 12.2,
         "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121", "chunk": 0},
    ])
    assert len(same_window) == 2
    across = review.dedupe_issues([
        {"category": "math", "severity": "minor", "start": 292.0, "end": 296.0,
         "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121", "chunk": 0},
        {"category": "math", "severity": "minor", "start": 292.5, "end": 296.5,
         "evidence": "公差写成 2，应为 3。", "suggested_fix": "改成 3", "chunk": 1},
    ])
    assert len(across) == 2
    assert {item["evidence"] for item in across} == {"A8 写成 120，应为 121。", "公差写成 2，应为 3。"}


def test_cross_chunk_dedupe_is_one_to_one_per_source_chunk():
    # 窗口 0 只有 A8；窗口 1 在重叠处同时报了 A8 和另一个很像的 A9。
    # 旧逻辑合并后只留一个 chunk，A9 会再并进去，三条变成一条。
    a8 = "黑板上 A8 写成 120，应为 121。"
    a8_again = "黑板把 A8 写成 120，应为 121。"
    a9 = "黑板上 A9 写成 130，应为 131。"
    merged = review.dedupe_issues([
        {"category": "math", "severity": "minor", "start": 14.0, "end": 18.0,
         "evidence": a8, "suggested_fix": "改成 121。", "chunk": 0},
        {"category": "math", "severity": "minor", "start": 14.2, "end": 18.2,
         "evidence": a8_again, "suggested_fix": "改成 121。", "chunk": 1},
        {"category": "math", "severity": "minor", "start": 16.0, "end": 20.0,
         "evidence": a9, "suggested_fix": "改成 131。", "chunk": 1},
    ])
    assert len(merged) == 2
    assert {item["evidence"] for item in merged} == {a8, a9}
    assert all("_chunks" not in item for item in merged)
    passed, violations = review.decide(
        merged, {"fail_severities": [], "max_counts": {"minor": 1}, "fail_categories": {}})
    assert passed is False and violations


def test_similar_cross_chunk_pair_does_not_hide_a_second_issue(tmp_path):
    first = json.dumps({
        "summary": "前一段里 A8 算错了。",
        "issues": [{
            "category": "math", "severity": "minor", "start": 14, "end": 18,
            "evidence": "黑板上 A8 写成 120，应为 121。", "suggested_fix": "改成 121。",
        }],
    }, ensure_ascii=False)
    second = json.dumps({
        "summary": "接缝处 A8 还在，旁边 A9 也错了。",
        "issues": [
            {"category": "math", "severity": "minor", "start": 2.2, "end": 6.2,
             "evidence": "黑板把 A8 写成 120，应为 121。", "suggested_fix": "改成 121。"},
            {"category": "math", "severity": "minor", "start": 4, "end": 8,
             "evidence": "黑板上 A9 写成 130，应为 131。", "suggested_fix": "改成 131。"},
        ],
    }, ensure_ascii=False)
    thresholds = {"fail_severities": [], "max_counts": {"minor": 1}, "fail_categories": {}}
    report, _, _ = _run(
        tmp_path, [first, second], duration=30, max_chunk=20, overlap=8, thresholds=thresholds)
    assert len(report["issues"]) == 2
    assert report["passed"] is False
    assert any(item["code"] == "RV001" for item in report["gate"]["issues"])
    assert {item["evidence"] for item in report["issues"]} == {
        "黑板上 A8 写成 120，应为 121。",
        "黑板上 A9 写成 130，应为 131。",
    }


def test_segment_end_tolerance_clamps_both_ends(tmp_path):
    parsed = review.parse_model_output(
        '{"summary": "段尾有一处。", "issues": [{"category": "math", "severity": "minor", '
        '"start": 10.04, "end": 10.04, "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121。"}]}',
        chunk_len=10,
    )
    issue = parsed["issues"][0]
    assert issue["start"] == 10 and issue["end"] == 10
    inside = review.parse_model_output(
        '{"summary": "段尾有一处。", "issues": [{"category": "math", "severity": "minor", '
        '"start": 9.5, "end": 10.04, "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121。"}]}',
        chunk_len=10,
    )
    assert inside["issues"][0]["start"] == 9.5 and inside["issues"][0]["end"] == 10
    with pytest.raises(review.ReviewSchemaError):
        review.parse_model_output(
            '{"summary": "段尾有一处。", "issues": [{"category": "math", "severity": "minor", '
            '"start": 10.06, "end": 10.06, "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121。"}]}',
            chunk_len=10,
        )

    first = json.dumps({
        "summary": "非最后一段的末尾多报了 40ms。",
        "issues": [{
            "category": "math", "severity": "minor", "start": 10.04, "end": 10.04,
            "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121。",
        }],
    }, ensure_ascii=False)
    second = json.dumps({"summary": "后一段没有问题。", "issues": []}, ensure_ascii=False)
    report, scripted, _ = _run(
        tmp_path, [first, second], duration=20, max_chunk=10, overlap=0,
        thresholds={"fail_severities": [], "max_counts": {}, "fail_categories": {}},
    )
    assert len(scripted.seen) == 2
    assert report["issues"][0]["start"] == 10 and report["issues"][0]["end"] == 10
    assert (tmp_path / "out" / "episode.review.json").exists()
    assert (tmp_path / "out" / "episode.review.md").exists()


def test_two_minor_issues_in_one_segment_still_trip_the_cap(tmp_path):
    text = json.dumps({
        "summary": "同一段里有两处不同的计算错误。",
        "issues": [
            {"category": "math", "severity": "minor", "start": 4, "end": 6,
             "evidence": "A8 写成 120，应为 121。", "suggested_fix": "改成 121。"},
            {"category": "math", "severity": "minor", "start": 4.2, "end": 6.4,
             "evidence": "公差写成 2，应为 3。", "suggested_fix": "改成 3。"},
        ],
    }, ensure_ascii=False)
    thresholds = {"fail_severities": [], "max_counts": {"minor": 1}, "fail_categories": {}}
    report, _, _ = _run(tmp_path, [text], duration=20, thresholds=thresholds)
    assert report["passed"] is False and len(report["issues"]) == 2
    assert any(item["code"] == "RV001" for item in report["gate"]["issues"])


def test_run_cost_ignores_lines_other_reviews_append(tmp_path):
    out = tmp_path / "out" / "episode.review.json"
    log = out.parent / "review_usage.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"run_id": "other", "est_usd": 50.0, "input_tokens": 9_000_000}) + "\n", encoding="utf-8")
    good = json.dumps({"summary": "这一段没有要退回的问题。", "issues": []}, ensure_ascii=False)

    class Racing(Scripted):
        def __call__(self, *, video, system, user, chunk):
            if chunk.index == 1:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"run_id": "racer", "est_usd": 50.0}) + "\n")
            return super().__call__(video=video, system=system, user=user, chunk=chunk)

    video = _video(tmp_path)
    report = review.review_episode(
        video, reviewer=Racing([good, good], [(1000, 100)]), out_json=out, model="gemini-4-argon",
        thresholds=review.load_thresholds(None), max_chunk_seconds=300, overlap_seconds=8,
        probe=lambda path: 400.0, cut=lambda src, start, end, dest: dest.write_bytes(b"clip"),
        today=WHEN, log=lambda *_: None,
    )
    assert report["usage"]["calls"] == 2
    assert report["usage"]["input_tokens"] == 2000 and report["usage"]["output_tokens"] == 200
    assert report["usage"]["est_usd"] == pytest.approx(0.006)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    ours = [row for row in rows if row.get("run_id") not in {"other", "racer"}]
    assert len(ours) == 2 and len({row["run_id"] for row in ours}) == 1
    assert review.total_usage(log)[1] == pytest.approx(100.006)


class _Usage:
    def __init__(self, prompt=0, candidates=0, thoughts=0, tool=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.thoughts_token_count = thoughts
        self.tool_use_prompt_token_count = tool


class _Stream:
    """按脚本吐出若干块，然后可选地在中途抛错。每次 generate_content_stream 消耗一幕。"""

    def __init__(self, scenes):
        self.scenes = list(scenes)

    def generate_content_stream(self, **kwargs):
        pieces, error = self.scenes.pop(0)
        yield from pieces
        if error is not None:
            raise error


def _gemini(stream, *, attempts=1):
    pytest.importorskip("google.genai")
    client = SimpleNamespace(models=stream, files=SimpleNamespace())
    return GeminiReviewer("gemini-4-argon", attempts=attempts, sleep=lambda _seconds: None, client=client)


def test_mid_stream_failure_keeps_partial_usage_and_logs_it(tmp_path):
    chunk = review.Chunk(0, 0.0, 10.0)
    video = _video(tmp_path)
    partial = SimpleNamespace(text="{", usage_metadata=_Usage(prompt=40, candidates=5, thoughts=1))
    reviewer = _gemini(_Stream([([partial], RuntimeError("503 mid stream"))]), attempts=1)
    with pytest.raises(review.ReviewError) as caught:
        reviewer(video=video, system="系统", user="用户", chunk=chunk)
    assert caught.value.calls == [(40, 6, "usage")]
    assert caught.value.token_source == "usage"

    out = tmp_path / "out" / "episode.review.json"
    failing = _gemini(_Stream([([partial], RuntimeError("503 mid stream"))]), attempts=1)
    with pytest.raises(review.ReviewError):
        review.review_episode(
            video, reviewer=failing, out_json=out, model="gemini-4-argon",
            thresholds=review.load_thresholds(None), probe=lambda path: 10.0, today=WHEN, log=lambda *_: None,
        )
    row = json.loads((out.parent / "review_usage.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert row["input_tokens"] == 40 and row["output_tokens"] == 6 and row["token_source"] == "usage"
    assert row["run_id"]


def test_stream_without_usage_is_estimated_not_confirmed_zero(tmp_path):
    chunk = review.Chunk(0, 0.0, 10.0)
    video = _video(tmp_path)
    expected = review.estimate_tokens("系统用户", chunk.length, "high")
    missing = _gemini(_Stream([([], RuntimeError("503 before any token"))]))
    with pytest.raises(review.ReviewError) as caught:
        missing(video=video, system="系统", user="用户", chunk=chunk)
    assert caught.value.calls == [(*expected, "estimate")]
    assert caught.value.token_source == "estimate"
    assert caught.value.calls != [(0, 0, "usage")]

    zeros = SimpleNamespace(text='{"summary":"看完了。","issues":[]}', usage_metadata=_Usage())
    estimated = _gemini(_Stream([([zeros], None)]))
    turn = estimated(video=video, system="系统", user="用户", chunk=chunk)
    assert turn.calls == [(*expected, "estimate")] and turn.token_source == "estimate"


def test_retry_keeps_partial_usage_from_the_failed_attempt(tmp_path):
    chunk = review.Chunk(0, 0.0, 10.0)
    partial = SimpleNamespace(text="{", usage_metadata=_Usage(prompt=40, candidates=5, thoughts=1))
    done = SimpleNamespace(
        text='{"summary":"看完了。","issues":[]}',
        usage_metadata=_Usage(prompt=100, candidates=8, thoughts=2),
    )
    reviewer = _gemini(_Stream([
        ([partial], RuntimeError("503 mid stream")),
        ([done], None),
    ]), attempts=2)
    turn = reviewer(video=_video(tmp_path), system="系统", user="用户", chunk=chunk)
    assert turn.calls == [(40, 6, "usage"), (100, 10, "usage")]
    assert turn.token_source == "usage"


@pytest.mark.parametrize("mode", ["poll", "failed", "timeout"])
def test_uploaded_file_is_deleted_when_processing_does_not_finish(tmp_path, monkeypatch, mode):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "sekret-key-xyz")
    monkeypatch.setattr("voice.review_gemini.INLINE_MAX_BYTES", 0)

    class Files:
        def __init__(self):
            self.deleted = []

        def upload(self, file):
            state = "FAILED" if mode == "failed" else "PROCESSING"
            return SimpleNamespace(name="files/to-delete", state=SimpleNamespace(name=state), uri="https://example/v")

        def get(self, name):
            if mode == "poll":
                raise RuntimeError("polling sekret-key-xyz")
            return SimpleNamespace(name=name, state=SimpleNamespace(name="PROCESSING"), uri="https://example/v")

        def delete(self, name):
            self.deleted.append(name)

    files = Files()
    client = SimpleNamespace(files=files, models=SimpleNamespace())
    reviewer = GeminiReviewer("gemini-4-argon", attempts=1, sleep=lambda _seconds: None, client=client)
    video = _video(tmp_path)
    with pytest.raises(review.ReviewError) as caught:
        reviewer(video=video, system="系统", user="用户", chunk=review.Chunk(0, 0.0, 4.0))
    assert files.deleted == ["files/to-delete"]
    message = str(caught.value)
    assert "sekret-key-xyz" not in message
    if mode == "poll":
        assert "等待视频处理失败" in message and "***" in message
    elif mode == "failed":
        assert "Gemini 处理视频文件失败" in message
    else:
        assert "超时" in message
