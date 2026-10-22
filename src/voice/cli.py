"""命令行：

    animind-voice synth SPEC --out 音频目录 --env 密钥文件 [--model M] [--voice V] [--speed 1.0] [--lang en]
    animind-voice align SPEC --audio 音频目录 [--backend mlx|faster] [--model 对齐模型] [--lang en]
    animind-voice pitch SPEC --audio 音频目录 [--fix --env 密钥文件] [--lang en]

synth 读 scene-spec 的 say，先用 zh-math-reading 展开公式、打印读法风险提示，再逐句合成并拼出 <beat>.wav + <beat>.timing.json；
align 用 Whisper 求每个书签的真实时刻，写回 timing.json，并报告哪些句子需要回听。
pitch 测每句的音高，找出像换了声音的句子；--fix 只重合成这些句子（同一段的其他句子沿用缓存），再重拼这一段。
--lang en 念英文版（每句的 en，书签写在 en 里；见 scene_spec.localize），三个命令要用同一个 --lang 和音频目录。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from scene_spec import localize
from scene_spec.loader import load_file

from . import script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="animind-voice")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("synth", help="逐句合成旁白")
    s.add_argument("spec", type=Path)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--env", type=Path, help="含 OPENROUTER_API_KEY / GEMINI_TTS_* 的文件；不会打印其内容")
    s.add_argument("--model", help="默认取环境变量 GEMINI_TTS_MODEL")
    s.add_argument("--voice", help="默认取环境变量 GEMINI_TTS_VOICE")
    s.add_argument("--speed", type=float, help="默认取 scene-spec 的 style.voice.speed，没有则 1.0")
    s.add_argument("--only", nargs="*", default=[], help="只处理这些 beat")
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
    args = parser.parse_args(argv)

    doc = localize(load_file(args.spec), args.lang)
    beats = [b for b in script.beats(doc) if not args.only or b.id in args.only]

    if args.command == "pitch":
        return pitch_command(doc, beats, args)

    if args.command == "synth":
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
        voice_style = doc.get("style", {}).get("voice", {})
        model = args.model or synth.default_model()
        voice = args.voice or synth.default_voice(model)
        if not model or not voice:
            print("缺少 TTS 模型或声音：用 --model/--voice，或在 --env 文件里设 GEMINI_TTS_MODEL / GEMINI_TTS_VOICE"
                  "（ElevenLabs：TTS_PROVIDER=elevenlabs、ELEVENLABS_VOICE_ID、ELEVENLABS_MODEL_ID）", file=sys.stderr)
            return 2
        os.environ["ELEVENLABS_VOICE_ID" if model.startswith("elevenlabs/") else "GEMINI_TTS_VOICE"] = voice
        speed = args.speed or float(voice_style.get("speed", 1.0))
        args.out.mkdir(parents=True, exist_ok=True)
        log_path = args.out / "tts_usage.jsonl"
        before, chars_before = synth.total_cost(log_path), synth.total_chars(log_path)
        for beat in beats:
            synth.synth_beat(beat, args.out, voice_style.get("persona", "teacher-hs"), model, voice, speed)
        after, chars_after = synth.total_cost(log_path), synth.total_chars(log_path)
        print(f"这次调用 TTS {after[0] - before[0]} 次，约 ${after[1] - before[1]:.4f}；累计 {after[0]} 次，约 ${after[1]:.4f}（{log_path}）")
        if chars_after > chars_before:
            print(f"其中按字符计费、单价未核的（ElevenLabs）：这次 {chars_after - chars_before} 字，累计 {chars_after} 字")
        return 0

    from . import align

    backend = args.backend or align.default_backend()
    model = args.model or align.DEFAULT_MODELS[backend]
    print(f"对齐后端 {backend}，模型 {model}")
    flagged = 0
    for beat in beats:
        results = align.align_beat(beat, args.audio, model, align.TRANSCRIBERS[backend])
        for r in results:
            if r.flagged:
                flagged += 1
                sure = f"；复核仍缺 {r.confirmed}" if r.confirmed else "（没有原始单句音频，未复核）"
                print(f"  ⚠ {beat.id}[{r.key}] 匹配 {r.ratio:.0%}；没听到 {r.missing}；多出 {r.extra}{sure}\n    听到：{r.heard}")
    print(f"完成；需要回听 {flagged} 句")
    return 0


def pitch_command(doc: dict, beats: list, args) -> int:
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
    speed = float(doc.get("style", {}).get("voice", {}).get("speed", 1.0))
    log_path = args.audio / "tts_usage.jsonl"
    before = synth.total_cost(log_path)
    for beat in beats:
        names = {f"{beat.id}-{s.index}" for s in beat.sentences} & set(bad)
        if not names:
            continue
        _, voice, model, _ = rows[sorted(names)[0]]
        os.environ["ELEVENLABS_VOICE_ID" if model.startswith("elevenlabs/") else "GEMINI_TTS_VOICE"] = voice
        synth.synth_beat(beat, args.audio, persona, model, voice, speed, force=names, loose=True)
    after = synth.total_cost(log_path)
    fixed = measure()
    still = [n for n in bad if fixed[n][3]]
    for name in bad:
        print(f"  {name}: {rows[name][0]:.0f} → {fixed[name][0]:.0f} Hz{'（仍异常）' if name in still else ''}")
    print(f"这次调用 TTS {after[0] - before[0]} 次，约 ${after[1] - before[1]:.4f}；重合成的段要重新 align，再回听新句子")
    return 1 if still else 0


if __name__ == "__main__":
    sys.exit(main())
