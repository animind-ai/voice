"""分句合成：每句一段音频，按语气留停顿拼成 beat 音频，写出 timing.json。

流程沿用 highSchool/lessons/sec6-7/kit/narrate.py：
- 模型名带 "/"（例如 google/gemini-3.8-flash-tts）走 OpenRouter，风格提示放在 instructions；否则走 Vertex；
  elevenlabs/<model_id>（例如 elevenlabs/eleven_v4）走 ElevenLabs，声音 ID 从 ELEVENLABS_VOICE_ID 读；
- 句间静音按句末标点（scene_spec.silence_after，say[].hold 可覆盖），开头留 0.3 秒，最后一句之后不留；
- speed ≠ 1 时用 ffmpeg atempo 只把人声变速（不变调，原声缓存不受影响），静音不随语速缩放，
  所以 timing.json 的 cues 之间的间隔就是 silence_after，scene_spec 的 SS487 能直接比。

缓存（见 ttscache.py）：每句的原声以 <音频目录>/.cache/tts/<sha256>.wav 为准，键是模型、声音、人设、语气、语言、
instructions 和送给 TTS 的正文。只改了一句时，同一段的其他句子直接读缓存，不调用 TTS，韵律也不变。
lines/<beat>-<下标>.wav（+ .json）是给 align 复核和 pitch 用的副本，每次合成按缓存重写，不再当缓存读——
只有两种例外：升级前合成的、说明完全对得上的副本会被收进缓存（不再花钱重念）；loose 模式（pitch --fix）
允许沿用人设文案改过之前的副本。

密钥只从环境变量读（--env 指向的文件会被加载进环境），从不打印。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from scene_spec import silence_after

from . import pitch, ttscache
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


ELEVENLABS_PREFIX = "elevenlabs/"
ELEVENLABS_DEFAULT_MODEL = "eleven_v4"


def is_elevenlabs(model: str | None) -> bool:
    return bool(model) and model.startswith(ELEVENLABS_PREFIX)


def elevenlabs_speak(instructions: str, text: str, model: str) -> bytes:
    """ElevenLabs（模型名写成 elevenlabs/<model_id>，例如 elevenlabs/eleven_v4）。

    声音从环境变量 ELEVENLABS_VOICE_ID 读（复刻音色的 ID 不进仓库），ELEVENLABS_STABILITY /
    ELEVENLABS_SIMILARITY_BOOST 可调；密钥 ELEVENLABS_API_KEY 由 SDK 自己从环境变量读。
    v4 没有风格提示，instructions 不送，但仍记进缓存键（换人设会重合成，和其他提供方一样）。
    返回 24 kHz 单声道 s16le PCM。需要 `uv sync --extra elevenlabs`。
    """
    from elevenlabs import ElevenLabs, VoiceSettings

    lang = "en" if text.isascii() else "zh"
    settings = VoiceSettings(stability=float(os.environ.get("ELEVENLABS_STABILITY", "0.5")),
                             similarity_boost=float(os.environ.get("ELEVENLABS_SIMILARITY_BOOST", "0.75")))
    client = ElevenLabs(timeout=180)
    with client.text_to_speech.with_raw_response.convert(
        text=text, voice_id=os.environ["ELEVENLABS_VOICE_ID"], model_id=model.removeprefix(ELEVENLABS_PREFIX),
        language_code=lang, output_format="pcm_24000", voice_settings=settings,
    ) as response:
        audio = b"".join(response.data)
    if not audio:
        raise RuntimeError("ElevenLabs 没有返回音频")
    return audio


def default_speak(instructions: str, text: str, model: str) -> bytes:
    if is_elevenlabs(model):
        return elevenlabs_speak(instructions, text, model)
    return openrouter_speak(instructions, text, model) if "/" in model else vertex_speak(instructions, text, model)


def voice_env(model: str) -> str:
    """speak 函数从哪个环境变量读声音：ElevenLabs 是 ELEVENLABS_VOICE_ID，其余是 GEMINI_TTS_VOICE。"""
    return "ELEVENLABS_VOICE_ID" if is_elevenlabs(model) else "GEMINI_TTS_VOICE"


def default_model(environ=None) -> str | None:
    """环境变量里的模型：TTS_PROVIDER=elevenlabs 时是 elevenlabs/<ELEVENLABS_MODEL_ID，默认 eleven_v4>，
    否则 GEMINI_TTS_MODEL。"""
    environ = os.environ if environ is None else environ
    if environ.get("TTS_PROVIDER", "").strip().lower() == "elevenlabs":
        return ELEVENLABS_PREFIX + (environ.get("ELEVENLABS_MODEL_ID") or ELEVENLABS_DEFAULT_MODEL)
    return environ.get("GEMINI_TTS_MODEL")


def default_voice(model: str | None, environ=None) -> str | None:
    """环境变量里这个模型对应的声音：ElevenLabs 用 ELEVENLABS_VOICE_ID，其余用 GEMINI_TTS_VOICE。"""
    environ = os.environ if environ is None else environ
    return environ.get(voice_env(model or ""))


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


def _speed(raw: Path, speed: float, cache: Path) -> Path:
    """变速后的音频，也在缓存里（键是原声字节的 sha256 + speed）；speed = 1 直接用原声。

    原声只读一次：算键的 sha256 和喂给 ffmpeg 的是同一份字节，不会把另一版原声的变速结果记到这一版的键下。
    """
    if speed == 1.0:
        return raw
    data = raw.read_bytes()
    # speed 用 repr：完整精度，1.250001 和 1.250002 是两个键（ffmpeg 拿到的 atempo 参数也不同）
    fields = {"kind": "speed", "source_sha256": hashlib.sha256(data).hexdigest(), "speed": repr(float(speed))}
    entry_key = ttscache.key(**fields)
    hit, _ = ttscache.lookup(cache, entry_key, RATE)
    if hit:
        return hit
    tmp = ttscache.scratch(cache, entry_key)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "wav", "-i", "pipe:0", "-filter:a", f"atempo={float(speed)!r}",
                    "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(tmp)], input=data, check=True)
    return ttscache.store(cache, entry_key, tmp, RATE, fields)


def instructions_for(persona: str, pace: str, lang: str = "zh") -> str:
    if lang == "en":
        return " ".join([PERSONAS_EN.get(persona, PERSONAS_EN["teacher-hs"]), PACE_EN[pace], VERBATIM_EN, STEADY_EN])
    return f"{PERSONAS.get(persona, PERSONAS['teacher-hs'])}{PACE[pace]}{VERBATIM}{STEADY}"


@dataclass
class Line:
    audio: Path  # 这句的原声：缓存里的不可变对象（loose 沿用 lines/ 旧副本时，是它的快照）
    view: Path  # lines/<name>.wav
    cached: bool  # 没有调用 TTS


def _view(src: Path, view: Path, stamp: dict) -> None:
    """把原声复制成 lines/<name>.wav，写上说明；内容没变就不动文件。"""
    if src != view and (not view.exists() or ttscache.sha256_file(view) != ttscache.sha256_file(src)):
        tmp = ttscache.temp_path(view)
        shutil.copyfile(src, tmp)
        os.replace(tmp, view)
    ttscache.write_text_atomic(view.with_suffix(".json"), json.dumps(stamp, ensure_ascii=False))


def _legacy(view: Path, stamp: dict, keys: tuple[str, ...]) -> bool:
    """lines/ 里的旧副本能不能用：说明里这些字段都对得上，音频也完好。"""
    stamp_path = view.with_suffix(".json")
    if not view.exists() or not stamp_path.exists():
        return False
    try:
        old = json.loads(stamp_path.read_text(encoding="utf-8"))
    except ValueError:
        return False
    return all(old.get(k) == stamp[k] for k in keys) and ttscache.wav_frames(view, RATE) is not None


def line_audio(lines_dir: Path, name: str, pace: str, text: str, persona: str, model: str, voice: str,
               speak: Speak = default_speak, log=print, force: bool = False, loose: bool = False, lang: str = "zh") -> Line:
    """一句的原声：缓存命中就读盘（不调用 TTS），否则合成、写进缓存。见 synth_line。"""
    lines_dir.mkdir(parents=True, exist_ok=True)
    cache = ttscache.tts_dir(lines_dir.parent)
    cache.mkdir(parents=True, exist_ok=True)
    view = lines_dir / f"{name}.wav"
    instructions = instructions_for(persona, pace, lang)
    fields = {"kind": "tts", "model": model, "voice": voice, "persona": persona, "pace": pace, "lang": lang,
              "instructions": instructions, "text": text}
    entry_key = ttscache.key(**fields)
    stamp = {"text": text, "instructions": instructions, "voice": voice, "model": model, "key": entry_key}
    if not force:
        hit, damaged = ttscache.lookup(cache, entry_key, RATE)
        if hit:
            _view(hit, view, stamp)
            return Line(hit, view, True)
        if damaged:
            log(f"  ⚠ {name} 的缓存坏了（{damaged}），重合成")
        elif _legacy(view, stamp, ("text", "instructions", "voice", "model")):  # 升级前的副本：收进缓存
            tmp = ttscache.scratch(cache, entry_key)
            shutil.copyfile(view, tmp)
            entry = ttscache.store(cache, entry_key, tmp, RATE, fields)
            _view(entry, view, stamp)
            return Line(entry, view, True)
        if loose and _legacy(view, stamp, ("text", "voice", "model")):
            # 人设文案改过之前的副本：不记到现在的键下（它不是用现在的 instructions 念的），只拍一个不可变快照来拼
            tmp = ttscache.scratch(cache, name)
            shutil.copyfile(view, tmp)
            return Line(ttscache.put_object(cache, tmp, RATE)[0], view, True)
    tmp = ttscache.scratch(cache, entry_key)
    best: tuple[float, bytes, float] | None = None
    for attempt in range(1, PITCH_RETRIES + 1):
        audio = speak(instructions, text, model)
        _to_wav(audio, tmp)
        seconds = wav_seconds(tmp)
        cost = log_usage(lines_dir.parent / "tts_usage.jsonl", name, model, text, instructions, seconds)
        f0 = pitch.median_f0(tmp)
        price = f"{len(text)} 字（单价未核）" if cost is None else f"约 ${cost:.4f}"
        log(f"  合成 {name} {seconds:.1f}s，音高 {f0:.0f} Hz，{price}")
        if not pitch.off_voice(f0, voice):
            best = None
            break
        if best is None or pitch.closeness(f0, voice) > best[0]:
            best = (pitch.closeness(f0, voice), audio, f0)
        log(f"  ⚠ {name} 音高 {f0:.0f} Hz，不像 {voice}（像换了声音），重合成（第 {attempt}/{PITCH_RETRIES} 次）")
    if best is not None:
        _to_wav(best[1], tmp)
        log(f"  ⚠ {name} 重试 {PITCH_RETRIES} 次仍像换了声音，留下最接近的一版（{best[2]:.0f} Hz），请回听")
    entry = ttscache.store(cache, entry_key, tmp, RATE, fields)
    _view(entry, view, stamp)
    return Line(entry, view, False)


def synth_line(lines_dir: Path, name: str, pace: str, text: str, persona: str, model: str, voice: str,
               speak: Speak = default_speak, log=print, force: bool = False, loose: bool = False, lang: str = "zh") -> Path:
    """合成一句（有缓存就直接用），返回 lines/<name>.wav。

    force：不读缓存，重合成并刷新缓存。loose：缓存没命中时，lines/ 里的旧副本只要文本、声音和模型对得上
    就沿用，不核对 instructions——只修个别句子时用，免得人设文案改过就把同一段的其他句子全部重念。
    """
    return line_audio(lines_dir, name, pace, text, persona, model, voice, speak, log, force, loose, lang).view


def estimate_cost(text: str, instructions: str, seconds: float, model: str = "") -> float | None:
    """估算一次调用的费用（美元）。ElevenLabs 按字符计费、套餐单价各异：没设 ELEVENLABS_USD_PER_CHAR 时返回 None，
    绝不套 Gemini 的价。"""
    if is_elevenlabs(model):
        rate = os.environ.get("ELEVENLABS_USD_PER_CHAR")
        return len(text) * float(rate) if rate else None
    input_tokens = (len(text) + len(instructions)) * 1.5  # 粗估：中文约 1–2 token/字（英文按字母算偏高，宁多不少）
    return input_tokens * PRICE_INPUT_PER_TOKEN + seconds * AUDIO_TOKENS_PER_SECOND * PRICE_OUTPUT_PER_TOKEN


def log_usage(path: Path, name: str, model: str, text: str, instructions: str, seconds: float) -> float | None:
    """每次真实调用 TTS 追加一行用量记录，返回估算费用（美元；单价未知时记 null、返回 None，字数见 total_chars）。"""
    import datetime

    cost = estimate_cost(text, instructions, seconds, model)
    record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "line": name, "model": model,
              "chars": len(text), "seconds": round(seconds, 2), "est_usd": None if cost is None else round(cost, 6)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return cost


def _usage_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def total_cost(path: Path) -> tuple[int, float]:
    """用量记录里的调用次数和估算总费用（单价未知、est_usd 为 null 的调用按 0 计，字数见 total_chars）。"""
    rows = _usage_rows(path)
    return len(rows), sum(r.get("est_usd") or 0.0 for r in rows)


def total_chars(path: Path) -> int:
    """单价未知（est_usd 为 null）的调用的字数之和：ElevenLabs 这类按字符计费的用它报用量。"""
    return sum(r.get("chars", 0) for r in _usage_rows(path) if r.get("est_usd") is None)


def synth_beat(beat: Beat, out_dir: Path, persona: str, model: str, voice: str, speed: float = 1.0,
               speak: Speak = default_speak, log=print, force: set[str] | None = None, loose: bool = False,
               no_cache: bool = False, reassemble: bool = False, before_publish: Callable[[], None] | None = None) -> dict:
    """合成一段：逐句查缓存，只给没命中的句子调用 TTS，再按 silence_after 拼成 <beat>.wav、写 timing.json。

    force：这些句子（<beat>-<index>）不读缓存，重合成并刷新缓存；no_cache：这一段的每一句都这样（CLI 的 --no-cache）。
    loose 见 synth_line。句子音频和停顿都和上次拼的时候一样，就沿用现有的 <beat>.wav 和 timing.json；
    reassemble（CLI 的 --force）照样重拼——句子仍然读缓存。
    沿用时 align 写进 timing.json 的书签时刻只在句子 key 和书签位置也没变时保留，否则清掉（要重新 align）。
    发布：before_publish()（CLI 用它在 manifest 里预写 pending）→ 预写 beats-next 说明 → 换 wav → 写 timing.json →
    写说明（wav 的 sha256、cues、model / voice / speed）。说明的 sha256 对不上 wav 就不会沿用，所以中途被打断下次一定重拼；
    manifest 怎么靠这两份说明恢复见 policy.py。
    这里不加锁：和别人共用音频目录时，调用方要持有 ttscache.dir_lock（CLI 的 synth / pitch --fix 整个运行都持有）。
    """
    if not beat.sentences:
        return {}
    cache = ttscache.tts_dir(out_dir)
    lines = [line_audio(out_dir / "lines", f"{beat.id}-{s.index}", s.pace, s.spoken, persona, model, voice, speak, log,
                        force=no_cache or f"{beat.id}-{s.index}" in (force or set()), loose=loose, lang=s.lang)
             for s in beat.sentences]
    parts = [_speed(line.audio, speed, cache) for line in lines]
    cached = sum(line.cached for line in lines)
    counts = f"{cached} cached, {len(lines) - cached} synthesized"
    gaps = [round(silence_after(s.text, s.hold), 3) if k + 1 < len(parts) else 0.0 for k, s in enumerate(beat.sentences)]
    # 来源也在指纹里：换了声音，哪怕音频字节碰巧一样，也重新发布、重新记来源
    fingerprint = ttscache.key(kind="beat", lead=LEAD, rate=RATE, gaps=gaps, parts=[ttscache.sha256_file(p) for p in parts],
                               model=model, voice=voice, speed=repr(float(speed)))
    # 对齐结果（marks / check / words）按句子 key 存，书签按它在读法文本里的位置求：这些变了，旧的对齐就作废
    script_fp = ttscache.key(kind="script", sentences=[[s.key, s.spoken, sorted(s.marks.items())] for s in beat.sentences])
    wav, timing_path = out_dir / f"{beat.id}.wav", out_dir / f"{beat.id}.timing.json"
    stamp_path = ttscache.beats_dir(out_dir) / f"{beat.id}.json"
    published = None if reassemble or no_cache else _published(wav, timing_path, stamp_path, fingerprint)
    if published is not None:
        stamp, timing = published
        if stamp.get("script") == script_fp:
            log(f"{beat.id}: {counts}；句子和停顿都没变，沿用 {wav.name} 和 {timing_path.name}（--force 重拼）")
            return timing
        timing = {"cues": timing["cues"], "total": timing["total"]}
        ttscache.write_text_atomic(timing_path, json.dumps(timing, ensure_ascii=False, indent=1))
        ttscache.write_text_atomic(stamp_path, json.dumps({**stamp, "script": script_fp}))
        log(f"{beat.id}: {counts}；音频没变，沿用 {wav.name}；句子 id 或书签位置变了，"
            f"清掉 {timing_path.name} 里过期的对齐结果（要重新 align）")
        return timing
    cues, cursor, inputs, filters = [], LEAD, [], []
    for k, (part, gap) in enumerate(zip(parts, gaps)):
        seconds = wav_seconds(part)
        cues.append({"start": round(cursor, 3), "end": round(cursor + seconds, 3)})
        inputs += ["-i", str(part)]
        filters.append(f"[{k}:a]apad=pad_dur={gap:.3f}[a{k}]")
        cursor += seconds + gap
    joined = "".join(f"[a{k}]" for k in range(len(parts)))
    filters.append(f"aevalsrc=0:d={LEAD}:s={RATE}[lead]")
    filters.append(f"[lead]{joined}concat=n={len(parts) + 1}:v=0:a=1[out]")
    tmp = ttscache.temp_path(wav)
    subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", "[out]",
                    "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", str(tmp)], check=True)
    timing = {"cues": cues, "total": round(cursor, 3)}
    stamp = {"fingerprint": fingerprint, "script": script_fp, "wav_sha256": ttscache.sha256_file(tmp), **timing,
             "model": model, "voice": voice, "speed": float(speed)}
    if before_publish is not None:
        before_publish()
    next_path = ttscache.next_dir(out_dir) / f"{beat.id}.json"
    ttscache.write_text_atomic(next_path, json.dumps(stamp))
    os.replace(tmp, wav)
    ttscache.write_text_atomic(timing_path, json.dumps(timing, ensure_ascii=False, indent=1))
    ttscache.write_text_atomic(stamp_path, json.dumps(stamp))
    next_path.unlink(missing_ok=True)
    log(f"{beat.id}: {counts}；{len(parts)} 句，{cursor:.1f}s")
    return timing


def _published(wav: Path, timing_path: Path, stamp_path: Path, fingerprint: str) -> tuple[dict, dict] | None:
    """已经发布的这一段还能不能沿用：说明里的指纹是这次的，wav 的 sha256 和 timing.json 的 cues / total 都和说明一致。
    能就返回 (说明, timing.json)。"""
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(stamp, dict) or not isinstance(timing, dict) or stamp.get("fingerprint") != fingerprint
            or not wav.exists() or stamp.get("wav_sha256") != ttscache.sha256_file(wav)
            or timing.get("cues") != stamp.get("cues") or timing.get("total") != stamp.get("total")):
        return None
    return stamp, timing
