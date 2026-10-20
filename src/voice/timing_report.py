"""时间报告（scene_spec.check_timing 消费的格式）：只含每句的 cues，由 synth 写出的 <beat>.timing.json 整理而来。

    {"beats": {"<beat id>": {"cues": [{"start": 0.3, "end": 2.1}, ...]}}}

actions（画面时刻）由 kit 产出，这里不写。没有 timing.json 的 beat 跳过（没合成就没有时刻）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .script import Beat

REPORT_NAME = "timing-report.json"


def build(beats: list[Beat], audio_dir: Path) -> dict:
    out = {}
    for beat in beats:
        path = audio_dir / f"{beat.id}.timing.json"
        if not path.exists():
            continue
        cues = json.loads(path.read_text(encoding="utf-8"))["cues"]
        out[beat.id] = {"cues": [{"start": c["start"], "end": c["end"]} for c in cues]}
    return {"beats": out}


def write(beats: list[Beat], audio_dir: Path) -> Path:
    path = audio_dir / REPORT_NAME
    path.write_text(json.dumps(build(beats, audio_dir), ensure_ascii=False, indent=1), encoding="utf-8")
    return path
