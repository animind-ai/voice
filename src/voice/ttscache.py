"""句级 TTS 缓存，以及音频目录的独占锁。

    .cache/objects/<sha256>.wav  音频对象（22050 Hz 单声道 s16le，就是 _to_wav 的输出），文件名就是内容的 sha256。
                                 不可变：同名文件只会是同样的内容，写入时已有完好的就不再动它
    .cache/tts/<key>.json        索引：键 → 对象（sha256、frames），以及算键用的字段（模型、声音、正文……）
    .cache/beats/<beat>.json     上次拼 <beat>.wav 用了哪些句子和停顿（指纹）、稿子的句子 key 和书签位置，
                                 以及发布出去的 wav 的 sha256、cues 和 model / voice / speed；没变就不重拼
    .cache/beats-next/<beat>.json 换 wav 之前预写的同样内容（manifest 被打断时靠它恢复来源，见 policy.py）
    .cache/synth.lock            目录锁（dir_lock）：synth 整个运行、align 写 timing.json、pitch --fix 都持有它

键是 sha256(FORMAT + 算键字段的 JSON)。原声（kind=tts）的字段：模型、声音、人设、语气、语言、送给 TTS 的
instructions 和正文；语速不在里面——speed 只是本地 atempo，原声不受影响。变速后的音频另存一个条目
（kind=speed，字段是原声字节的 sha256 + speed），所以改语速只重算 atempo，不重新调用 TTS。

命中要同时满足：索引读得出、format 和 key 对得上、对象存在、它的 sha256 就是索引记的、wav 头合法
（单声道 16 位、采样率对、数据长度和帧数一致）。任何一项不满足都当作损坏，重新生成。
刷新同一个键（--no-cache、pitch --fix）只是写一个新对象、原子替换索引：已经拿到旧对象路径的读者读到的仍是完整的旧音频。
写入一律先写同目录的临时文件再 os.replace，先写对象再写索引：中途被打断只会留下没有索引的对象（不算命中）。
旧对象不会自动删除；删掉 .cache/ 就是清空缓存。改了缓存的格式或算键规则就把 FORMAT 加一，旧条目自然失效。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows：没有 flock，不加锁（voice 只在 macOS / Linux 上跑）
    fcntl = None

FORMAT = 2
DIR = ".cache"
LOCK_NAME = "synth.lock"


class Busy(RuntimeError):
    """--no-wait 时音频目录正被另一个 synth（或 align / pitch --fix）占用。"""


def tts_dir(audio_dir: Path) -> Path:
    return audio_dir / DIR / "tts"


def beats_dir(audio_dir: Path) -> Path:
    return audio_dir / DIR / "beats"


def next_dir(audio_dir: Path) -> Path:
    """发布前预写的说明（这次要发布的 wav 的 sha256 + 来源），见 policy.py 的说明。"""
    return audio_dir / DIR / "beats-next"


def _objects(root: Path) -> Path:
    return root.parent / "objects"


def key(**fields) -> str:
    payload = json.dumps({"format": FORMAT, **fields}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temp_path(path: Path) -> Path:
    """path 同目录下的临时文件名（后缀保持 .wav 之类，ffmpeg 靠它认格式）。"""
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def wav_frames(path: Path, rate: int) -> int | None:
    """合法的单声道 16 位 rate Hz wav 返回帧数；头坏了、被截断、格式不对返回 None。"""
    try:
        with wave.open(str(path)) as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != rate:
                return None
            n = w.getnframes()
            if n <= 0 or len(w.readframes(n)) != 2 * n:
                return None
            return n
    except (OSError, EOFError, wave.Error):
        return None


def scratch(root: Path, stem: str) -> Path:
    """在对象目录里给即将生成的音频取一个临时文件名（和对象同一个文件系统，os.replace 才是原子的）。"""
    objects = _objects(root)
    objects.mkdir(parents=True, exist_ok=True)
    return temp_path(objects / f"{stem}.wav")


def put_object(root: Path, produced: Path, rate: int) -> tuple[Path, str, int]:
    """把 produced（用 scratch 取名）收成不可变对象，返回 (对象路径, sha256, 帧数)。
    同样内容的对象已经在、而且完好，就丢掉 produced，不碰已有的文件。"""
    frames = wav_frames(produced, rate)
    if frames is None:
        raise RuntimeError(f"生成的音频不是合法的 {rate} Hz 单声道 wav：{produced}")
    digest = sha256_file(produced)
    obj = _objects(root) / f"{digest}.wav"
    if obj.exists() and sha256_file(obj) == digest:
        produced.unlink()
    else:  # 没有，或者坏了：同名就该是这份内容，换上去就是修好
        os.replace(produced, obj)
    return obj, digest, frames


def lookup(root: Path, entry_key: str, rate: int) -> tuple[Path | None, str | None]:
    """(命中的对象, None)；没有条目是 (None, None)；条目坏了是 (None, 原因)。"""
    index = root / f"{entry_key}.json"
    if not index.exists():
        return None, None
    try:
        meta = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "索引不是 JSON"
    if not isinstance(meta, dict) or meta.get("format") != FORMAT or meta.get("key") != entry_key:
        return None, "索引的格式版本或键对不上"
    obj = _objects(root) / f"{meta.get('sha256')}.wav"
    if not obj.exists():
        return None, "音频对象不见了"
    if sha256_file(obj) != meta.get("sha256"):
        return None, "音频对象的 sha256 和文件名不一致（被截断或改过）"
    if wav_frames(obj, rate) != meta.get("frames"):
        return None, "音频不是合法的 wav 或帧数不对"
    return obj, None


def store(root: Path, entry_key: str, produced: Path, rate: int, fields: dict) -> Path:
    """produced 收成对象，再原子地把键指向它（刷新同一个键也只是换索引，旧对象原样留着）。返回对象路径。"""
    obj, digest, frames = put_object(root, produced, rate)
    meta = {**fields, "format": FORMAT, "key": entry_key, "frames": frames, "sha256": digest}
    write_text_atomic(root / f"{entry_key}.json", json.dumps(meta, ensure_ascii=False, indent=1))
    return obj


@contextmanager
def dir_lock(audio_dir: Path, wait: bool = True, log=print) -> Iterator[None]:
    """独占音频目录（<音频目录>/.cache/synth.lock，flock，进程退出自动释放）。

    synth 整个运行都持有它（读写缓存、发布 wav / timing / 说明、写 manifest 和时间报告），align 写 timing.json、
    pitch --fix 也持有它；只读的 review 不用。已被占用时 wait 就打一行日志等着，否则抛 Busy。不可重入。
    """
    path = audio_dir / DIR / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if not wait:
                    raise Busy(f"{audio_dir} 正被另一个 synth（或 align / pitch --fix）使用（{path}）；"
                               "去掉 --no-wait 就会等它跑完") from None
                log(f"waiting for another synth on {audio_dir}（或 align / pitch --fix；--no-wait 不等直接退出）")
                fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
