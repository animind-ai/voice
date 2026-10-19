# animind-voice

[中文](#中文) | [English](#english)

## 中文

[scene-spec](https://github.com/animind-ai/scene-spec) 的旁白流水线：读场景文件里的 `say`，逐句合成旁白，再用 Whisper 求出每个书签 `{@名字}` 的真实时刻，顺带检查 TTS 有没有漏念、多念。输出直接给 [kit](https://github.com/animind-ai/kit) 渲染用。

### 两个命令

```bash
uv sync --extra align        # macOS（Apple 芯片）；Linux + NVIDIA 用 uv sync --extra align-cuda
# 1. 合成：每句一段音频（按文本 + 语气 + 人设 + 模型缓存），拼成 <beat>.wav + <beat>.timing.json
uv run animind-voice synth ../scene-spec/examples/s67-p01.scene.yaml --out build/audio --env /path/to/.env
# 2. 对齐 + 回听：把书签的真实时刻写回 timing.json，列出需要人去听的句子
uv run animind-voice align ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio
# 3. 音高检查：找出像换了声音的句子；--fix 只重合成这些句子，之后对这几段重新 align
uv run animind-voice pitch ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio [--fix --env /path/to/.env]
```

Gemini TTS 偶尔会在单句里换成另一个性别的声音（同一个 `voice`、同一个模型）。所以人设文案不写性别，每句都要求“全程保持同一个声音”；合成完还会测这句的基频中位数，和这个声音应有的音区比（目前认得 Kore、Aoede、Leda、Zephyr 是女声，低于 150 Hz 算换声；Puck、Charon、Fenrir、Orus 是男声，高于 190 Hz 算换声），换了声就重合成，最多 3 次，每次都记账；3 次都不过关时留最接近的一版并提示回听。其他声音不做判断。`pitch` 对已经合成好的目录做同样的检查。

英文版：三个命令都加 `--lang en`，念每句的 `en`（书签写在 `en` 里，见 scene-spec 设计稿 §7.7），用单独的音频目录。合成指令换成英文、Vertex 用 en-US，不过中文读法规则；对齐时 Whisper 按英文听写，听到的阿拉伯数字先拼成单词（2 → two）再和稿子配对。

`synth` 的模型和声音从 `--env` 文件或环境变量读：`GEMINI_TTS_MODEL`（名字带 `/` 走 OpenRouter，需要 `OPENROUTER_API_KEY`；否则走 Vertex，需要 `uv sync --extra vertex` 和 `GOOGLE_*`）、`GEMINI_TTS_VOICE`。密钥从不打印，也不要放进仓库。人设取场景文件的 `style.voice.persona`，语速取 `style.voice.speed`（atempo 变速，不变调）。

合成之前，每句旁白先交给 [zh-math-reading](https://github.com/animind-ai/zh-math-reading)：`say` 里的 `$...$` 公式展开成读法（`$a_{n+1}^2$` → “a n 加一的平方”），书签位置按展开后的文字计算；同时打印读法风险提示（冒号后紧跟字母、长句、多音字），只提示、不拦截。不含公式的句子读法逐字节不变，已有的音频缓存继续有效。zh-math-reading 是同级目录的可编辑路径依赖，和 scene-spec 一样要并排检出。

`align` 在本机跑 `whisper-large-v3-turbo`，不需要联网 API。后端按平台自动选：macOS 用 [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（Apple 芯片，`--extra align`），其他平台用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper)（CTranslate2，有 CUDA 就用 GPU、float16，`--extra align-cuda`）；`--backend mlx|faster` 或环境变量 `ANIMIND_ALIGN_BACKEND` 可以强制指定，`ANIMIND_ALIGN_DEVICE=cpu|cuda` 指定设备，多卡时用 `CUDA_VISIBLE_DEVICES` 选卡。两个后端用的是同一个模型的不同格式（`mlx-community/whisper-large-v3-turbo` / `mobiuslabsgmbh/faster-whisper-large-v3-turbo`），书签时刻相差在 0.1 秒左右。做法：

1. 按 timing.json 的句子起止把 beat 音频切开，每句单独听写，稿子作为提示词；
2. Whisper 给出逐字时间戳，和稿子逐字配对（数字按汉字比较，标点不算）；
3. 书签落在哪个字前面，就取那个字开始的时刻；没配上的字在前后配上的字之间插值；
4. Whisper 偶尔在半个多字节字符处断掉、后面的字没有时间戳，这时从断点往后补听。

结果写回 timing.json：

```json
{"cues": [...], "total": 27.03,
 "marks": {"s1": {"d0": 5.01}, "s2": {"d1": 8.04, "d2": 9.74, "d3": 12.0}},
 "check": {"s1": {"ratio": 1.0, "missing": [], "extra": [], "heard": "..."}},
 "aligner": "mlx-community/whisper-large-v3-turbo"}
```

kit 读到 `marks` 就用真实时刻，没有就按字符位置比例估算。

匹配率低于 85%，或者漏念 / 多念两个字以上的句子会被标出来，例如 §6.7 第 01 支里：

```
⚠ Problem[s4] 匹配 95%；没听到 ['后面']；多出 []
  听到：然后告诉我们，数列 A n 是二、五、十一、二十一、三十六，省略，而且，……
```

### 开发

```bash
uv run pytest      # 不调用 TTS 和 Whisper：合成用假的 speak，对齐用假的 transcriber
ruff check .
```

### 作者与许可

作者：Boxi Yu（[@BoxiYu](https://github.com/BoxiYu)）

代码以 [MIT 许可](LICENSE)发布。

## English

The narration pipeline for [scene-spec](https://github.com/animind-ai/scene-spec): reads the `say` lines in a scene file, synthesizes narration sentence by sentence, then uses Whisper to find the real time of every bookmark `{@name}`, checking along the way whether the TTS skipped or added words. The output goes straight to [kit](https://github.com/animind-ai/kit) for rendering.

### Two commands

```bash
uv sync --extra align        # macOS (Apple silicon); on Linux + NVIDIA use uv sync --extra align-cuda
# 1. Synthesize: one audio clip per sentence (cached by text + tone + persona + model), joined into <beat>.wav + <beat>.timing.json
uv run animind-voice synth ../scene-spec/examples/s67-p01.scene.yaml --out build/audio --env /path/to/.env
# 2. Align + listen-back: write bookmarks' real times back into timing.json and list sentences a human should listen to
uv run animind-voice align ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio
# 3. Pitch check: find sentences that sound like a different voice; --fix re-synthesizes only those, then re-run align on them
uv run animind-voice pitch ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio [--fix --env /path/to/.env]
```

Gemini TTS occasionally switches to a voice of the other gender within a single sentence (same `voice`, same model). So persona text doesn't specify gender, and every sentence asks to "keep the same voice throughout"; after synthesis, the sentence's median fundamental frequency is measured and compared against the range expected for that voice (currently Kore, Aoede, Leda, Zephyr are recognized as female voices, with below 150 Hz counted as a switch; Puck, Charon, Fenrir, Orus as male voices, with above 190 Hz counted as a switch). If the voice switched, the sentence is re-synthesized, up to 3 times, each attempt logged; if all 3 fail, the closest take is kept and flagged for listen-back. Other voices are not judged. `pitch` runs the same check on an already-synthesized directory.

English version: add `--lang en` to all three commands to read each sentence's `en` (bookmarks are written inside `en`; see scene-spec design doc §7.7), using a separate audio directory. Synthesis instructions switch to English and Vertex uses en-US, skipping the Chinese reading rules; during alignment, Whisper transcribes in English, and heard Arabic numerals are first spelled out as words (2 → two) before matching against the script.

`synth` reads the model and voice from the `--env` file or environment variables: `GEMINI_TTS_MODEL` (a name containing `/` goes through OpenRouter and needs `OPENROUTER_API_KEY`; otherwise Vertex, which needs `uv sync --extra vertex` and `GOOGLE_*`), and `GEMINI_TTS_VOICE`. Keys are never printed and should never be put in the repository. The persona comes from the scene file's `style.voice.persona`, and the speed from `style.voice.speed` (atempo time-stretching, pitch unchanged).

Before synthesis, every narration sentence is first passed to [zh-math-reading](https://github.com/animind-ai/zh-math-reading): `$...$` formulas in `say` are expanded into their spoken readings (`$a_{n+1}^2$` → "a n 加一的平方"), and bookmark positions are computed on the expanded text; reading-risk hints are printed at the same time (a letter right after a colon, long sentences, polyphonic characters) — hints only, nothing is blocked. Sentences without formulas are byte-for-byte unchanged, so existing audio caches remain valid. zh-math-reading is an editable path dependency in a sibling directory and, like scene-spec, must be checked out alongside.

`align` runs `whisper-large-v3-turbo` locally, with no online API needed. The backend is chosen automatically by platform: on macOS, [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple silicon, `--extra align`); on other platforms, [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, GPU with float16 if CUDA is available, `--extra align-cuda`). `--backend mlx|faster` or the environment variable `ANIMIND_ALIGN_BACKEND` forces a backend, `ANIMIND_ALIGN_DEVICE=cpu|cuda` sets the device, and with multiple GPUs use `CUDA_VISIBLE_DEVICES` to pick one. The two backends use different formats of the same model (`mlx-community/whisper-large-v3-turbo` / `mobiuslabsgmbh/faster-whisper-large-v3-turbo`), and bookmark times differ by about 0.1 s. How it works:

1. Cut the beat audio at the sentence boundaries in timing.json and transcribe each sentence separately, with the script as the prompt;
2. Whisper gives per-character timestamps, which are matched character by character against the script (numbers compared as Chinese characters, punctuation ignored);
3. A bookmark takes the start time of the character it precedes; unmatched characters are interpolated between the nearest matched ones;
4. Whisper occasionally cuts off in the middle of a multi-byte character, leaving later characters without timestamps; in that case, transcription resumes from the break point.

Results are written back into timing.json:

```json
{"cues": [...], "total": 27.03,
 "marks": {"s1": {"d0": 5.01}, "s2": {"d1": 8.04, "d2": 9.74, "d3": 12.0}},
 "check": {"s1": {"ratio": 1.0, "missing": [], "extra": [], "heard": "..."}},
 "aligner": "mlx-community/whisper-large-v3-turbo"}
```

When kit finds `marks`, it uses the real times; otherwise it estimates proportionally from character positions.

Sentences with a match rate below 85%, or with more than two characters skipped / added, are flagged, for example in §6.7 episode 01 (the tool prints in Chinese: "匹配 95%；没听到 ['后面']；多出 []" = "match 95%; not heard ['后面']; extra []", "听到" = "heard"):

```
⚠ Problem[s4] 匹配 95%；没听到 ['后面']；多出 []
  听到：然后告诉我们，数列 A n 是二、五、十一、二十一、三十六，省略，而且，……
```

### Development

```bash
uv run pytest      # no calls to TTS or Whisper: synthesis uses a fake speak, alignment a fake transcriber
ruff check .
```

### Author and license

Author: Boxi Yu ([@BoxiYu](https://github.com/BoxiYu))

The code is released under the [MIT License](LICENSE).
