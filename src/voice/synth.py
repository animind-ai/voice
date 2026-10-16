"""分句合成：每句一段音频（按文本 + 语气 + 人设 + 模型缓存），按语气留停顿拼成 beat 音频，写出 timing.json。

流程沿用 highSchool/lessons/sec6-7/kit/narrate.py：
- 模型名带 "/"（例如 google/gemini-3.8-flash-tts）走 OpenRouter，风格提示放在 instructions；否则走 Vertex；
- 句间停顿按语气（normal / brisk / slow / ask），开头留 0.3 秒；
- speed ≠ 1 时用 ffmpeg atempo 变速（不变调），原声缓存不受影响。

密钥只从环境变量读（--env 指向的文件会被加载进环境），从不打印。
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from .audio import wav_seconds
from .script import Beat

PERSONAS = {
    "teacher-hs": "你是一位亲切、有经验的高中数学老师，正对着屏幕上的题目给学生讲解。说话自然，像面对面讲题，不要播音腔，也不要念稿子的感觉。",
    "peer-ms": "你是一位好奇、耐心的学长，正和初中同学一起看一道数学题。语气轻松自然，适度设问，不要播音腔。",
    "friend-ps": "你是小学生的大朋友，正一起看一个数学小问题。句子短、语速慢、语气亲切，多用“我们一起看”。",
}
PACE = {
    "normal": "正常语速，语调自然起伏。",
    "brisk": "这几句是简单的口算，语速轻快一些，一带而过。",
    "slow": "这是重点，语速比平时稍慢一点，关键词加重，但不要拖沓。",
    "ask": "带着设问的语气，像在引导学生一起想，问句尾音上扬。",
}
GAP = {"normal": 0.35, "brisk": 0.25, "slow": 0.6, "ask": 0.55}
LEAD = 0.3
RATE = 22050

Speak = Callable[[str, str, str], bytes]  # (instructions, text, model) → 音频字节（wav 或 24k s16le PCM）


def load_env(path: Path | None) -> None:
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def openrouter_speak(instructions: str, text: str, model: str) -> bytes:
    import urllib.request

    body = {"model": model, "voice": os.environ["GEMINI_TTS_VOICE"], "input": text,
            "instructions": instructions, "response_format": "pcm"}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def vertex_speak(instructions: str, text: str, model: str) -> bytes:
    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"), api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=f"{instructions}请用这种方式朗读下面这段话：\n{text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code="zh-CN",
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=os.environ["GEMINI_TTS_VOICE"])),
            ),
        ),
    )
    for candidate in getattr(response, "candidates", None) or []:
        for part in getattr(getattr(candidate, "content", None), "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data
    raise RuntimeError("Gemini-TTS 没有返回音频")


def default_speak(instructions: str, text: str, model: str) -> bytes:
    return openrouter_speak(instructions, text, model) if "/" in model else vertex_speak(instructions, text, model)


def _to_wav(audio: bytes, out: Path) -> None:
    raw = out.with_suffix(".bin")
    raw.write_bytes(audio)
    is_wav = audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if not is_wav:
        cmd += ["-f", "s16le", "-ar", "24000", "-ac", "1"]
    cmd += ["-i", str(raw), "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(out)]
    subprocess.run(cmd, check=True)
    raw.unlink()


def _speed(part: Path, speed: float) -> Path:
    if abs(speed - 1.0) < 1e-3:
        return part
    fast = part.with_name(f"{part.stem}@{speed:g}.wav")
    if not fast.exists() or fast.stat().st_mtime < part.stat().st_mtime:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(part), "-filter:a", f"atempo={speed}",
                        "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(fast)], check=True)
    return fast


def synth_line(lines_dir: Path, name: str, pace: str, text: str, persona: str, model: str, voice: str,
               speak: Speak = default_speak, log=print) -> Path:
    lines_dir.mkdir(parents=True, exist_ok=True)
    wav = lines_dir / f"{name}.wav"
    stamp_path = wav.with_suffix(".json")
    instructions = f"{PERSONAS.get(persona, PERSONAS['teacher-hs'])}{PACE[pace]}"
    stamp = {"text": text, "instructions": instructions, "voice": voice, "model": model}
    if wav.exists() and stamp_path.exists() and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
        return wav
    _to_wav(speak(instructions, text, model), wav)
    stamp_path.write_text(json.dumps(stamp, ensure_ascii=False), encoding="utf-8")
    log(f"  合成 {name} {wav_seconds(wav):.1f}s")
    return wav


def synth_beat(beat: Beat, out_dir: Path, persona: str, model: str, voice: str, speed: float = 1.0,
               speak: Speak = default_speak, log=print) -> dict:
    if not beat.sentences:
        return {}
    parts = [
        _speed(synth_line(out_dir / "lines", f"{beat.id}-{s.index}", s.pace, s.spoken, persona, model, voice, speak, log), speed)
        for s in beat.sentences
    ]
    cues, cursor, inputs, filters = [], LEAD, [], []
    for k, (s, part) in enumerate(zip(beat.sentences, parts)):
        seconds = wav_seconds(part)
        cues.append({"start": round(cursor, 3), "end": round(cursor + seconds, 3)})
        gap = GAP[s.pace] / speed if k + 1 < len(parts) else 0.0
        inputs += ["-i", str(part)]
        filters.append(f"[{k}:a]apad=pad_dur={gap:.3f}[a{k}]")
        cursor += seconds + gap
    joined = "".join(f"[a{k}]" for k in range(len(parts)))
    filters.append(f"aevalsrc=0:d={LEAD}:s={RATE}[lead]")
    filters.append(f"[lead]{joined}concat=n={len(parts) + 1}:v=0:a=1[out]")
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                    "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(out_dir / f"{beat.id}.wav")], check=True)
    timing = {"cues": cues, "total": round(cursor, 3)}
    (out_dir / f"{beat.id}.timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{beat.id}: {len(parts)} 句，{cursor:.1f}s")
    return timing
