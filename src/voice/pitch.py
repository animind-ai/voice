"""音高检查：Gemini TTS 偶尔会在单句里换成另一个性别的声音（同一个 voice、同一个模型）。

每句测浊音段的基频中位数（自相关法），和这个声音应有的音区比。只认得下面列出的声音；
没列出的声音不做绝对判断。
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

# Gemini 预置声音里我们确认过性别的几个。女声低于 FEMALE_FLOOR、男声高于 MALE_CEILING 就算换了声。
FEMALE = {"Kore", "Aoede", "Leda", "Zephyr"}
MALE = {"Puck", "Charon", "Fenrir", "Orus"}
FEMALE_FLOOR = 150.0
MALE_CEILING = 190.0


def load(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr, n, channels, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype={2: np.int16, 4: np.int32}[width]).astype(np.float32)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    return (x / peak if peak else x), sr


def median_f0(path: Path) -> float:
    """浊音帧（自相关峰 > 0.5）的基频中位数，单位 Hz；听不到浊音时返回 0。"""
    x, sr = load(path)
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for start in range(0, len(x) - frame, hop):
        seg = x[start:start + frame]
        if np.sqrt((seg ** 2).mean()) < 0.02:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.5:
            f0s.append(sr / lag)
    return float(np.median(f0s)) if f0s else 0.0


def off_voice(f0: float, voice: str) -> bool:
    """这句的音高是不是明显不像这个声音（换了性别）。"""
    if not f0:
        return False
    if voice in FEMALE:
        return f0 < FEMALE_FLOOR
    if voice in MALE:
        return f0 > MALE_CEILING
    return False


def closeness(f0: float, voice: str) -> float:
    """越大越像这个声音；重试几次都不过关时，留最像的一版。"""
    if voice in FEMALE:
        return f0
    if voice in MALE:
        return -f0
    return 0.0
