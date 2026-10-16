"""命令行：

    animind-voice synth SPEC --out 音频目录 --env 密钥文件 [--model M] [--voice V] [--speed 1.0]
    animind-voice align SPEC --audio 音频目录 [--model 对齐模型]

synth 读 scene-spec 的 say，先用 zh-math-reading 展开公式、打印读法风险提示，再逐句合成并拼出 <beat>.wav + <beat>.timing.json；
align 用 Whisper 求每个书签的真实时刻，写回 timing.json，并报告哪些句子需要回听。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    a.add_argument("--model", default=None)
    a.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args(argv)

    doc = load_file(args.spec)
    beats = [b for b in script.beats(doc) if not args.only or b.id in args.only]

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
        model = args.model or os.environ.get("GEMINI_TTS_MODEL")
        voice = args.voice or os.environ.get("GEMINI_TTS_VOICE")
        if not model or not voice:
            print("缺少 TTS 模型或声音：用 --model/--voice，或在 --env 文件里设 GEMINI_TTS_MODEL / GEMINI_TTS_VOICE", file=sys.stderr)
            return 2
        os.environ["GEMINI_TTS_VOICE"] = voice
        speed = args.speed or float(voice_style.get("speed", 1.0))
        args.out.mkdir(parents=True, exist_ok=True)
        log_path = args.out / "tts_usage.jsonl"
        before = synth.total_cost(log_path)
        for beat in beats:
            synth.synth_beat(beat, args.out, voice_style.get("persona", "teacher-hs"), model, voice, speed)
        after = synth.total_cost(log_path)
        print(f"这次调用 TTS {after[0] - before[0]} 次，约 ${after[1] - before[1]:.4f}；累计 {after[0]} 次，约 ${after[1]:.4f}（{log_path}）")
        return 0

    from . import align

    flagged = 0
    for beat in beats:
        results = align.align_beat(beat, args.audio, args.model or align.DEFAULT_MODEL)
        for r in results:
            if r.flagged:
                flagged += 1
                sure = f"；复核仍缺 {r.confirmed}" if r.confirmed else "（没有原始单句音频，未复核）"
                print(f"  ⚠ {beat.id}[{r.key}] 匹配 {r.ratio:.0%}；没听到 {r.missing}；多出 {r.extra}{sure}\n    听到：{r.heard}")
    print(f"完成；需要回听 {flagged} 句")
    return 0


if __name__ == "__main__":
    sys.exit(main())
