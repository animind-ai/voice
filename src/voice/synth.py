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

from . import pitch
from .audio import wav_seconds
from .script import Beat

PERSONAS = {
    "teacher-hs": "你是一位亲切、有经验的高中数学老师，正对着屏幕上的题目给学生讲解。说话自然，像面对面讲题，不要播音腔，也不要念稿子的感觉。",
    "peer-ms": "你是一位好奇、耐心的高年级同学，正和初中同学一起看一道数学题。语气轻松自然，适度设问，不要播音腔。",
    "friend-ps": "你是小学生的大朋友，正一起看一个数学小问题。句子短、语速慢、语气亲切，多用“我们一起看”。",
}
PACE = {
    "normal": "正常语速，语调自然起伏。",
    "brisk": "这几句是简单的口算，语速轻快一些，一带而过。",
    "slow": "这是重点，语速比平时稍慢一点，关键词加重，但不要拖沓。",
    "ask": "带着设问的语气，像在引导学生一起想，问句尾音上扬。",
}
# 每句都加：Gemini TTS 常在逗号后、问句前插进稿子里没有的"诶""嗯""哎"（洞察 #3 #4 #5 #11 #15 #21 都遇到过）。
# 改这一句会让所有句子的缓存失效（缓存按 instructions 记），已经出片的集不受影响，除非重新合成。
VERBATIM = "严格照稿子念，一个字不加、一个字不漏，不要加“诶”“嗯”“哎”这类稿子里没有的语气词。"
# Gemini TTS 偶尔会在单句里换成另一个性别的声音（同一个 voice）；人设里不写性别，再明确要求声音前后一致。
# 合成后还会测音高（pitch.py），换了声就重合成。
STEADY = "全程保持同一个声音，音色、音高前后一致，不要换声音。"
PITCH_RETRIES = 3
# 英文版（style.lang: en）：同样的人设和要求，用英文写给 TTS。
PERSONAS_EN = {
    "teacher-hs": "You are a warm, experienced high-school maths teacher explaining the problem on screen to a student. "
                  "Speak naturally, as if face to face. Not a newsreader voice, and never sound like you are reading a script.",
    "peer-ms": "You are a curious, patient older student looking at a maths problem together with a younger student. "
               "Relaxed and natural, not a newsreader voice.",
    "friend-ps": "You are a big friend to a primary-school child, looking at a small maths puzzle together. "
                 "Short sentences, slow and warm.",
}
PACE_EN = {
    "normal": "Normal pace, with natural intonation.",
    "brisk": "This is simple arithmetic: a little brisker, move through it lightly.",
    "slow": "This is the key point: slightly slower than usual, stress the key words, but don't drag.",
    "ask": "A guiding, questioning tone, as if inviting the student to think along; let questions rise at the end.",
}
VERBATIM_EN = "Read the script exactly as written: add nothing, drop nothing, no filler words like \"um\", \"uh\" or \"so\" that aren't in it."
STEADY_EN = "Keep the same voice throughout: same timbre and pitch from start to finish."
LANGUAGE_CODE = {"zh": "zh-CN", "en": "en-US"}
GAP = {"normal": 0.35, "brisk": 0.25, "slow": 0.6, "ask": 0.55}
# 单价（美元）：OpenRouter 上 google/gemini-3.8-flash-tts 输入 $0.5/百万 token、输出 $9/百万 token；
# Gemini 音频输出按 25 token/秒计，约 $0.0135/分钟。价格变了就改这里。
PRICE_INPUT_PER_TOKEN = 0.5e-6
PRICE_OUTPUT_PER_TOKEN = 9e-6
AUDIO_TOKENS_PER_SECOND = 25
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

    lang = "en" if text.isascii() else "zh"  # 英文版的稿子全是西文字符
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                          location=os.environ.get("GOOGLE_CLOUD_REGION", "us-central1"), api_key=os.environ["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model=model,
        contents=(f"{instructions}请用这种方式朗读下面这段话：\n{text}" if lang == "zh"
                  else f"{instructions} Read the following in that way:\n{text}"),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=LANGUAGE_CODE[lang],
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


def elevenlabs_speak(instructions: str, text: str, model: str) -> bytes:
    """ElevenLabs（模型名写成 elevenlabs/<model_id>，例如 elevenlabs/eleven_v4）。

    声音从环境变量 ELEVENLABS_VOICE_ID 读（复刻音色的 ID 不进仓库）；v4 没有风格提示，instructions 不送，
    但仍记进缓存戳（换人设不会重合成，用 --loose 的语义）。返回 24 kHz 单声道 PCM。
    """
    from elevenlabs import ElevenLabs, VoiceSettings

    lang = "en" if text.isascii() else "zh"
    settings = VoiceSettings(stability=float(os.environ.get("ELEVENLABS_STABILITY", "0.5")),
                             similarity_boost=float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.75")))
    client = ElevenLabs(timeout=180)  # ELEVENLABS_API_KEY 由 SDK 自己从环境变量读
    with client.text_to_speech.with_raw_response.convert(
        text=text, voice_id=os.environ["ELEVENLABS_VOICE_ID"], model_id=model.removeprefix("elevenlabs/"),
        language_code=lang, output_format="pcm_24000", voice_settings=settings,
    ) as response:
        audio = b"".join(response.data)
    if not audio:
        raise RuntimeError("ElevenLabs 没有返回音频")
    return audio


def default_speak(instructions: str, text: str, model: str) -> bytes:
    if model.startswith("elevenlabs/"):
        return elevenlabs_speak(instructions, text, model)
    return openrouter_speak(instructions, text, model) if "/" in model else vertex_speak(instructions, text, model)


def default_voice(model: str | None) -> str | None:
    """环境变量里这个模型对应的声音名：ElevenLabs 用 ELEVENLABS_VOICE_ID，其余用 GEMINI_TTS_VOICE。"""
    if model and model.startswith("elevenlabs/"):
        return os.environ.get("ELEVENLABS_VOICE_ID")
    return os.environ.get("GEMINI_TTS_VOICE")


def default_model() -> str | None:
    """环境变量里的模型：TTS_PROVIDER=elevenlabs 时是 elevenlabs/<ELEVENLABS_MODEL_ID>，否则 GEMINI_TTS_MODEL。"""
    if os.environ.get("TTS_PROVIDER") == "elevenlabs":
        return f"elevenlabs/{os.environ.get('ELEVENLABS_MODEL_ID', 'eleven_v4')}"
    return os.environ.get("GEMINI_TTS_MODEL")


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


def instructions_for(persona: str, pace: str, lang: str = "zh") -> str:
    if lang == "en":
        return " ".join([PERSONAS_EN.get(persona, PERSONAS_EN["teacher-hs"]), PACE_EN[pace], VERBATIM_EN, STEADY_EN])
    return f"{PERSONAS.get(persona, PERSONAS['teacher-hs'])}{PACE[pace]}{VERBATIM}{STEADY}"


def synth_line(lines_dir: Path, name: str, pace: str, text: str, persona: str, model: str, voice: str,
               speak: Speak = default_speak, log=print, force: bool = False, loose: bool = False, lang: str = "zh") -> Path:
    """合成一句（有缓存就直接用）。

    force：不管缓存，重合成。loose：缓存只核对文本、声音和模型，不核对 instructions——
    只修个别句子时用，免得人设文案改过就把同一段的其他句子全部重念。
    """
    lines_dir.mkdir(parents=True, exist_ok=True)
    wav = lines_dir / f"{name}.wav"
    stamp_path = wav.with_suffix(".json")
    instructions = instructions_for(persona, pace, lang)
    stamp = {"text": text, "instructions": instructions, "voice": voice, "model": model}
    if not force and wav.exists() and stamp_path.exists():
        cached = json.loads(stamp_path.read_text(encoding="utf-8"))
        keys = ("text", "voice", "model") if loose else tuple(stamp)
        if all(cached.get(k) == stamp[k] for k in keys):
            return wav
    best: tuple[float, bytes, float] | None = None
    for attempt in range(1, PITCH_RETRIES + 1):
        audio = speak(instructions, text, model)
        _to_wav(audio, wav)
        seconds = wav_seconds(wav)
        cost = log_usage(lines_dir.parent / "tts_usage.jsonl", name, model, text, instructions, seconds)
        f0 = pitch.median_f0(wav)
        price = f"约 ${cost:.4f}" if not model.startswith("elevenlabs/") else f"{len(text)} 字"
        log(f"  合成 {name} {seconds:.1f}s，音高 {f0:.0f} Hz，{price}")
        if not pitch.off_voice(f0, voice):
            best = None
            break
        if best is None or pitch.closeness(f0, voice) > best[0]:
            best = (pitch.closeness(f0, voice), audio, f0)
        log(f"  ⚠ {name} 音高 {f0:.0f} Hz，不像 {voice}（像换了声音），重合成（第 {attempt}/{PITCH_RETRIES} 次）")
    if best is not None:
        _to_wav(best[1], wav)
        log(f"  ⚠ {name} 重试 {PITCH_RETRIES} 次仍像换了声音，留下最接近的一版（{best[2]:.0f} Hz），请回听")
    stamp_path.write_text(json.dumps(stamp, ensure_ascii=False), encoding="utf-8")
    return wav


def estimate_cost(text: str, instructions: str, seconds: float, model: str = "") -> float | None:
    """估算一次调用的费用（美元）。ElevenLabs 按字符计费、套餐单价各异：没设 ELEVENLABS_USD_PER_CHAR 时记 None，
    绝不套 Gemini 的价。"""
    if model.startswith("elevenlabs/"):
        rate = os.environ.get("ELEVENLABS_USD_PER_CHAR")
        return len(text) * float(rate) if rate else None
    input_tokens = (len(text) + len(instructions)) * 1.5  # 粗估：中文约 1–2 token/字（英文按字母算偏高，宁多不少）
    return input_tokens * PRICE_INPUT_PER_TOKEN + seconds * AUDIO_TOKENS_PER_SECOND * PRICE_OUTPUT_PER_TOKEN


def log_usage(path: Path, name: str, model: str, text: str, instructions: str, seconds: float) -> float:
    """每次真实调用 TTS 追加一行用量记录，返回估算费用（美元；单价未知时记 None、返回 0）。"""
    import datetime

    cost = estimate_cost(text, instructions, seconds, model)
    record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "line": name, "model": model,
              "chars": len(text), "seconds": round(seconds, 2), "est_usd": None if cost is None else round(cost, 6)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cost or 0.0


def total_cost(path: Path) -> tuple[int, float]:
    """用量记录里的调用次数和估算总费用（单价未知的调用按 0 计，字符数见 total_chars）。"""
    if not path.exists():
        return 0, 0.0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows), sum(r.get("est_usd") or 0.0 for r in rows)


def total_chars(path: Path) -> int:
    """用量记录里单价未知（est_usd 为 null）的调用的字符数之和，ElevenLabs 这类按字符计费的用它报用量。"""
    if not path.exists():
        return 0
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sum(r.get("chars", 0) for r in rows if r.get("est_usd") is None)


def synth_beat(beat: Beat, out_dir: Path, persona: str, model: str, voice: str, speed: float = 1.0,
               speak: Speak = default_speak, log=print, force: set[str] | None = None, loose: bool = False) -> dict:
    """force：这些句子（<beat>-<index>）不管缓存重合成；loose 见 synth_line。"""
    if not beat.sentences:
        return {}
    parts = [
        _speed(synth_line(out_dir / "lines", f"{beat.id}-{s.index}", s.pace, s.spoken, persona, model, voice, speak, log,
                          force=f"{beat.id}-{s.index}" in (force or set()), loose=loose, lang=s.lang), speed)
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
