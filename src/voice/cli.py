"""命令行：

    animind-voice synth SPEC --out 音频目录 --env 密钥文件 [--model M] [--voice V] [--speed 1.0] [--allow-env-voice] [--lang en]
                        [--only BEAT ...] [--force] [--no-cache] [--no-wait]
    animind-voice align SPEC --audio 音频目录 [--backend mlx|faster] [--model 对齐模型] [--lang en]
    animind-voice pitch SPEC --audio 音频目录 [--fix --env 密钥文件] [--lang en]
    animind-voice review EPISODE.mp4 [--spec SPEC] [--audio 音频目录] [--chapters EPISODE.chapters.json] [--env 密钥文件] [--model M] [--out 报告.json]

synth 读 scene-spec 的 say，先用 zh-math-reading 展开公式、打印读法风险提示，再逐句合成并拼出 <beat>.wav + <beat>.timing.json，
每句的原声缓存在 <音频目录>/.cache/（只改一句只重念这一句，日志 "Hook: 2 cached, 1 synthesized"；--no-cache 全部重念），
写出 manifest.json（实际用的 model / voice / speed）和 timing-report.json（给 scene-spec check --timing 用）。
声音以场景的 style.voice 为准；--model/--voice/--speed 显式覆盖；GEMINI_TTS_* 环境变量只有 --allow-env-voice 或 VOICE_ALLOW_ENV_OVERRIDE=1 才采用；
align 用 Whisper 求每个书签的真实时刻，写回 timing.json，并报告哪些句子需要回听。
pitch 测每句的音高，找出像换了声音的句子；--fix 只重合成这些句子（同一段的其他句子沿用缓存），再重拼这一段。
review 把成片（可带场景稿和 timing）交给 Gemini 看画面和声音，写出 JSON 报告和一份短 Markdown。退出码：0 通过，1 未通过，2 用法或调用失败。
synth / pitch --fix 开头把每段的 wav、timing.json、发布说明对一遍（policy.heal）：上次发布被打断的按说明修好；
wav 和哪份说明都对不上的段不进时间报告，manifest 记成来源不明，synth 以 1 退出（除非这次重合成了它）；align 拒绝对齐这种段（退出码 1）。
synth / align / pitch --fix 整个运行独占音频目录（.cache/synth.lock）：另一个在跑就等（打一行 waiting for another synth on …），--no-wait 则直接退出 2。
--lang en 念英文版（每句的 en，书签写在 en 里；见 scene_spec.localize）。synth / align / pitch 要用同一个 --lang 和音频目录；review 用它读场景稿。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scene_spec import localize
from scene_spec.loader import load_file
from scene_spec.quality import REQUIRED_MODEL, REQUIRED_SPEED, REQUIRED_VOICE

from . import policy, script, timing_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animind-voice")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="逐句合成旁白")
    s.add_argument("spec", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--env", type=Path, help="含 OPENROUTER_API_KEY / GEMINI_TTS_* 的文件；不会打印其内容")
    s.add_argument("--model", help="覆盖场景 style.voice.model（会提示）；默认按场景，没写则 google/gemini-3.8-flash-tts")
    s.add_argument("--voice", help="覆盖场景 style.voice.voice（会提示）；默认按场景，没写则 Kore")
    s.add_argument("--speed", type=float, help="覆盖场景 style.voice.speed（会提示）；默认按场景，没写则 1.0")
    s.add_argument("--allow-env-voice", action="store_true",
                   help="允许 GEMINI_TTS_MODEL / GEMINI_TTS_VOICE 覆盖场景的声音（会大声警告）；也可设 VOICE_ALLOW_ENV_OVERRIDE=1")
    s.add_argument("--only", nargs="*", default=[], help="只处理这些 beat")
    s.add_argument("--force", action="store_true",
                   help="句子和停顿都没变也重拼 <beat>.wav、重写 timing.json；没改的句子仍读缓存，不重新调用 TTS")
    s.add_argument("--no-cache", action="store_true", help="不读句子缓存：每句都重新调用 TTS，并刷新缓存（隐含 --force）")
    a = sub.add_parser("align", help="Whisper 对齐书签 + 回听检查")
    a.add_argument("spec", type=Path)
    a.add_argument("--audio", type=Path, required=True)
    a.add_argument("--backend", choices=["mlx", "faster"], help="听写后端；默认 macOS 用 mlx，其他平台用 faster(-whisper)，或取 ANIMIND_ALIGN_BACKEND")
    a.add_argument("--model", default=None, help="默认按后端取 whisper-large-v3-turbo 的对应版本")
    a.add_argument("--only", nargs="*", default=[])
    p = sub.add_parser("pitch", help="逐句测音高，找出像换了声音的句子；--fix 只重合成这些句子")
    p.add_argument("spec", type=Path)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--fix", action="store_true", help="重合成异常句（要 --env），之后要重新 align")
    p.add_argument("--env", type=Path)
    p.add_argument("--only", nargs="*", default=[])
    for cmd in (s, a, p):
        cmd.add_argument("--lang", choices=["zh", "en"], default="zh", help="念哪种语言的版本，默认中文")
        cmd.add_argument("--no-wait", action="store_true",
                         help="音频目录正被另一个 synth / align / pitch --fix 使用时不等，直接报错退出（默认等它跑完）")
    r = sub.add_parser("review", help="用 Gemini 看整集成片，写出审片报告")
    r.add_argument("video", type=Path, help="成片，通常是 mp4")
    r.add_argument("--spec", type=Path, help="这一集的 scene-spec；有的话会把旁白和书签附给模型")
    r.add_argument("--audio", type=Path, help="synth/align 的输出目录，读取 <beat>.timing.json")
    r.add_argument("--chapters", type=Path,
                   help="kit 写的 <成片>.chapters.json（默认取视频旁边的同名文件）；有 --spec 时用它把问题对到场景文件的行")
    r.add_argument("--env", type=Path, help="含 GEMINI_API_KEY 或 GOOGLE_* 的文件；不会打印其内容")
    r.add_argument("--model", help="默认 GEMINI_REVIEW_MODEL，否则 gemini-4-argon")
    r.add_argument("--out", type=Path, help="报告 JSON 的路径，或一个目录；默认写在视频旁边")
    r.add_argument("--thresholds", type=Path, help="JSON，覆盖通过/失败阈值")
    r.add_argument("--max-chunk-seconds", type=float, default=300, help="超过这个长度就切段，默认 300 秒")
    r.add_argument("--overlap-seconds", type=float, default=8, help="相邻段重叠的秒数，默认 8")
    r.add_argument("--media-resolution", choices=["low", "high"], default="high", help="数学画面默认 high，便于读字")
    r.add_argument("--media-processing", choices=["static", "agentic"], default="static",
                   help="static 逐秒看画面；agentic 更省 token，可能漏掉布局问题")
    r.add_argument("--attempts", type=int, default=3, help="遇到 429 或 5xx 时一次请求最多试几次")
    r.add_argument("--prompt-version", default="v1", help="提示词目录的版本，默认 v1")
    r.add_argument("--lang", choices=["zh", "en"], default="zh", help="场景稿用哪种语言，默认中文")
    args = parser.parse_args(argv)

    if args.command == "review":
        return review_command(args)

    doc = localize(load_file(args.spec), args.lang)
    try:
        all_beats = script.beats(doc)
    except ValueError as exc:
        print(f"{args.spec}: {exc}", file=sys.stderr)
        return 2
    beats = [b for b in all_beats if not args.only or b.id in args.only]

    from .ttscache import Busy

    try:
        if args.command == "pitch":
            return pitch_command(doc, beats, args)
        if args.command == "synth":
            return synth_command(doc, all_beats, beats, args)
        return align_command(doc, beats, args)
    except Busy as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


def _locked(audio_dir: Path, args):
    """这个音频目录的独占锁（ttscache.dir_lock）。写音频目录的命令整个运行都持有它；只读的 review 不用。"""
    from .ttscache import dir_lock

    return dir_lock(audio_dir, wait=not args.no_wait, log=lambda line: print(line, file=sys.stderr))


def synth_command(doc: dict, all_beats: list, beats: list, args) -> int:
    from . import synth

    warned = 0
    for beat in beats:
        for sentence in beat.sentences:
            for w in sentence.warnings():
                warned += 1
                print(f"  ⚠ {beat.id}[{sentence.key}] {w}")
    if warned:
        print(f"读法检查：{warned} 条提示（只提示，不拦截合成；改 say 文本后会重新合成这一句）")

    synth.load_env(args.env)
    try:
        choice = policy.resolve(doc, model=args.model, voice=args.voice, speed=args.speed,
                                allow_env=policy.env_override_allowed(os.environ, args.allow_env_voice))
    except ValueError as exc:
        print(f"{args.spec}: {exc}", file=sys.stderr)
        return 2
    for line in choice.notices:
        print(f"提示：{line}", file=sys.stderr)
    for line in choice.warnings:
        print(line, file=sys.stderr)
    if not choice.policy_ok:
        print(f"WARNING: 这批音频不符合发布政策（{'；'.join(choice.policy_problems())}）。"
              f"发布要求 {REQUIRED_MODEL}、{REQUIRED_VOICE}、语速 {REQUIRED_SPEED:g}；这批输出不可发布。", file=sys.stderr)
    # speak 函数从 GEMINI_TTS_VOICE 读声音（OpenRouter 的 body / Vertex 的 voice_name），这里填入实际用的那个
    os.environ["GEMINI_TTS_VOICE"] = choice.voice
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "tts_usage.jsonl"
    persona = doc.get("style", {}).get("voice", {}).get("persona", "teacher-hs")
    # 整个运行独占音频目录：句子缓存、wav / timing / 说明、manifest、时间报告都只有这一个 synth 在写
    with _locked(args.out, args):
        _heal(args.out, {b.id for b in beats})  # 上次在发布途中被打断：按说明把 timing.json、说明、manifest 修好
        before = synth.total_cost(log_path)
        for beat in beats:
            # 换 wav 之前先在 manifest 里把这一段记成 pending（不合规），发布完再提交实际来源
            synth.synth_beat(beat, args.out, persona, choice.model, choice.voice, choice.speed,
                             speak=synth.default_speak, no_cache=args.no_cache, reassemble=args.force,
                             before_publish=lambda b=beat.id: policy.mark_pending(args.out, b, args.lang, choice.sources))
            if (args.out / f"{beat.id}.timing.json").exists():
                policy.write_manifest(args.out, choice, args.lang, [beat.id])
        after = synth.total_cost(log_path)
        done = [b.id for b in beats if (args.out / f"{b.id}.timing.json").exists()]
        conflicts = policy.write_manifest(args.out, choice, args.lang, done)  # 提醒按最终状态给
        report = timing_report.write(all_beats, args.out)  # 对不上 wav 的 beat 不收（见 timing_report.py）
        broken = policy.broken_beats(args.out)
    for conflict in conflicts:
        print(f"WARNING: {args.out}：{conflict}", file=sys.stderr)
    print(f"这次调用 TTS {after[0] - before[0]} 次，约 ${after[1] - before[1]:.4f}；累计 {after[0]} 次，约 ${after[1]:.4f}（{log_path}）")
    print(f"模型 {choice.model}，声音 {choice.voice}，语速 {choice.speed:g}；时间报告 {report}")
    if broken:
        print(f"错误：{args.out} 里这些 beat 的音频和 timing.json / 发布说明对不上，没写进时间报告：{'、'.join(broken)}。"
              f"请重合成：animind-voice synth {args.spec} --out {args.out} --only {' '.join(broken)}", file=sys.stderr)
        return 1
    return 0


def _heal(audio_dir: Path, redo: set[str]) -> dict[str, str]:
    """运行开头核对每段的 wav、timing.json、说明（policy.heal），打印修了什么；返回修不好的 beat。redo 是这次会重合成的。"""
    repaired, changed, broken = policy.heal(audio_dir)
    for line in repaired:
        print(f"按发布说明修复：{line}", file=sys.stderr)
    for line in changed:
        print(f"manifest 按发布说明修正：{line}", file=sys.stderr)
    for beat_id, problem in broken.items():
        if beat_id in redo:
            print(f"提示：{beat_id}：{problem}；这次会重合成它", file=sys.stderr)
        else:
            print(f"错误：{beat_id}：{problem}。manifest 记成来源不明，时间报告不收它；"
                  f"请重合成这一段（synth --only {beat_id}）", file=sys.stderr)
    return broken


def align_command(doc: dict, beats: list, args) -> int:
    from . import align, trace

    if not args.audio.is_dir():
        print(f"找不到音频目录 {args.audio}，先运行 synth", file=sys.stderr)
        return 2
    backend = args.backend or align.default_backend()
    model = args.model or align.DEFAULT_MODELS[backend]
    print(f"对齐后端 {backend}，模型 {model}")
    refs = trace.scene_refs(doc, args.spec, args.lang)
    flagged, refused = 0, []
    with _locked(args.audio, args):  # align 把书签时刻写回 timing.json：不能和 synth 同时改
        for beat in beats:
            try:
                results = align.align_beat(beat, args.audio, model, align.TRANSCRIBERS[backend])
            except align.Inconsistent as exc:  # wav / timing / 说明对不上：不对齐这一段，其他段照做
                print(f"错误：{exc}", file=sys.stderr)
                refused.append(beat.id)
                continue
            flagged += sum(r.flagged for r in results)
            for line in align.warnings(results, refs, beat.id):
                print(f"  ⚠ {line}")
    print(f"完成；需要回听 {flagged} 句" + (f"；没有对齐 {'、'.join(refused)}" if refused else ""))
    return 1 if refused else 0


def review_command(args) -> int:
    from . import review, synth

    synth.load_env(args.env)
    model = args.model or os.environ.get("GEMINI_REVIEW_MODEL") or review.DEFAULT_MODEL
    out_json = _review_out(args.video, args.out)
    try:
        thresholds = review.load_thresholds(args.thresholds)
        reviewer = review.default_reviewer(
            model, resolution=args.media_resolution, processing=args.media_processing, attempts=args.attempts)
        log_path = out_json.parent / "review_usage.jsonl"
        report = review.review_episode(
            args.video, reviewer=reviewer, out_json=out_json, model=model, spec=args.spec, audio=args.audio, chapters=args.chapters,
            lang=args.lang, thresholds=thresholds, max_chunk_seconds=args.max_chunk_seconds,
            overlap_seconds=args.overlap_seconds, prompt_version=args.prompt_version,
        )
    except review.ReviewError as exc:
        print(review.redact(str(exc)), file=sys.stderr)
        log_path = out_json.parent / "review_usage.jsonl"
        if log_path.exists():
            calls, usd = review.total_usage(log_path)
            print(f"已经记下来的调用在 {log_path}（累计 {calls} 次，约 ${usd:.4f}）", file=sys.stderr)
        return 2
    after = review.total_usage(out_json.parent / "review_usage.jsonl")
    status = "通过" if report["passed"] else "未通过"
    print(f"{args.video.name}: {status}（{report['usage']['calls']} 次模型调用，约 ${report['usage']['est_usd']:.4f}）")
    usage = report["usage"]
    print(f"这次调用 {usage['calls']} 次，约 ${usage['est_usd']:.4f}；累计 {after[0]} 次，约 ${after[1]:.4f}（{log_path}）")
    print(out_json.with_suffix(".md").read_text(encoding="utf-8"))
    return 0 if report["passed"] else 1


def _review_out(video: Path, out: Path | None) -> Path:
    if out is None:
        return video.with_name(video.stem + ".review.json")
    if out.suffix.lower() == ".json":
        return out
    return out / f"{video.stem}.review.json"


def pitch_command(doc: dict, beats: list, args) -> int:
    if args.fix:  # 要重合成、重拼：整个过程独占音频目录（测音高也在锁里，看到的是一致的一组）
        with _locked(args.audio, args):
            _heal(args.audio, set())
            try:
                code = _pitch(doc, beats, args)
            finally:  # 重拼的段按说明提交来源（和当初一样的 model / voice / speed）
                policy.reconcile(args.audio)
            broken = policy.broken_beats(args.audio)
        if broken:
            print(f"错误：{args.audio} 里这些 beat 的音频和 timing.json / 发布说明对不上：{'、'.join(broken)}。"
                  f"请用 synth --only {' '.join(broken)} 重合成", file=sys.stderr)
            return code or 1
        return code
    return _pitch(doc, beats, args)


def _pitch(doc: dict, beats: list, args) -> int:
    import json

    from . import pitch, synth

    def measure() -> dict[str, tuple[float, str, str, bool]]:
        rows = {}
        for beat in beats:
            for s in beat.sentences:
                name = f"{beat.id}-{s.index}"
                wav, stamp = args.audio / "lines" / f"{name}.wav", args.audio / "lines" / f"{name}.json"
                if not wav.exists() or not stamp.exists():
                    continue
                meta = json.loads(stamp.read_text(encoding="utf-8"))
                f0 = pitch.median_f0(wav)
                rows[name] = (f0, meta["voice"], meta["model"], pitch.off_voice(f0, meta["voice"]))
        return rows

    rows = measure()
    if not rows:
        print(f"{args.audio} 里没有合成好的句子", file=sys.stderr)
        return 2
    f0s = sorted(r[0] for r in rows.values() if r[0])
    bad = [n for n, r in rows.items() if r[3]]
    print(f"{len(rows)} 句，音高中位 {f0s[len(f0s) // 2]:.0f} Hz，像换了声音的 {len(bad)} 句")
    for name in bad:
        print(f"  ⚠ {name} {rows[name][0]:.0f} Hz（声音 {rows[name][1]}）")
    if not bad or not args.fix:
        return 1 if bad else 0

    synth.load_env(args.env)
    persona = doc.get("style", {}).get("voice", {}).get("persona", "teacher-hs")
    default_speed = float(doc.get("style", {}).get("voice", {}).get("speed", 1.0))
    log_path = args.audio / "tts_usage.jsonl"
    before = synth.total_cost(log_path)
    for beat in beats:
        names = {f"{beat.id}-{s.index}" for s in beat.sentences} & set(bad)
        if not names:
            continue
        _, voice, model, _ = rows[sorted(names)[0]]
        os.environ["GEMINI_TTS_VOICE"] = voice
        # 重合成要用这个 beat 当初实际的语速，不是场景里写的
        speed = policy.beat_speed(args.audio, beat.id, default_speed)
        synth.synth_beat(beat, args.audio, persona, model, voice, speed, force=names, loose=True,
                         before_publish=lambda b=beat.id: policy.mark_pending(args.audio, b))
    after = synth.total_cost(log_path)
    fixed = measure()
    still = [n for n in bad if fixed[n][3]]
    for name in bad:
        print(f"  {name}: {rows[name][0]:.0f} → {fixed[name][0]:.0f} Hz{'（仍异常）' if name in still else ''}")
    print(f"这次调用 TTS {after[0] - before[0]} 次，约 ${after[1] - before[1]:.4f}；重合成的段要重新 align，再回听新句子")
    return 1 if still else 0


if __name__ == "__main__":
    sys.exit(main())
