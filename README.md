# animind-voice

[scene-spec](https://github.com/animind-ai/scene-spec) 的旁白流水线：读场景文件里的 `say`，逐句合成旁白，再用 Whisper 求出每个书签 `{@名字}` 的真实时刻，顺带检查 TTS 有没有漏念、多念。输出直接给 [kit](https://github.com/animind-ai/kit) 渲染用。成片出来之后，`review` 把视频交给 Gemini 看画面和声音，写出可以当门槛用的审片报告。

## 命令

```bash
uv sync --extra align        # macOS（Apple 芯片）；Linux + NVIDIA 用 uv sync --extra align-cuda
# 1. 合成：每句一段音频（缓存在 build/audio/.cache/，见「句级缓存」），拼成 <beat>.wav + <beat>.timing.json
uv run animind-voice synth ../scene-spec/examples/s67-p01.scene.yaml --out build/audio --env /path/to/.env
# 2. 对齐 + 回听：把书签的真实时刻写回 timing.json，列出需要人去听的句子
uv run animind-voice align ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio
# 3. 音高检查：找出像换了声音的句子；--fix 只重合成这些句子，之后对这几段重新 align
uv run animind-voice pitch ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio [--fix --env /path/to/.env]
# 4. 审片：看整集成片（画面 + 声音）。有场景稿和 timing 就一并附上
uv sync --extra review
uv run animind-voice review build/s67-p01.mp4 \
  --spec ../scene-spec/examples/s67-p01.scene.yaml \
  --audio build/audio \
  --env /path/to/.env \
  --out build/review/s67-p01.review.json
```

Gemini TTS 偶尔会在单句里换成另一个性别的声音（同一个 `voice`、同一个模型）。所以人设文案不写性别，每句都要求“全程保持同一个声音”；合成完还会测这句的基频中位数，和这个声音应有的音区比（目前认得 Kore、Aoede、Leda、Zephyr 是女声，低于 150 Hz 算换声；Puck、Charon、Fenrir、Orus 是男声，高于 190 Hz 算换声），换了声就重合成，最多 3 次，每次都记账；3 次都不过关时留最接近的一版并提示回听。其他声音不做判断。`pitch` 对已经合成好的目录做同样的检查。

英文版：synth、align、pitch 都加 `--lang en`，念每句的 `en`（书签写在 `en` 里，见 scene-spec 设计稿 §7.7），用单独的音频目录。合成指令换成英文、Vertex 用 en-US，不过中文读法规则；对齐时 Whisper 按英文听写，听到的阿拉伯数字先拼成单词（2 → two）再和稿子配对。`review` 的 `--lang en` 只影响附给模型的场景稿。

`synth` 的模型和声音从 `--env` 文件或环境变量读：`GEMINI_TTS_MODEL`（名字带 `/` 走 OpenRouter，需要 `OPENROUTER_API_KEY`；否则走 Vertex，需要 `uv sync --extra vertex` 和 `GOOGLE_*`）、`GEMINI_TTS_VOICE`。密钥从不打印，也不要放进仓库。人设取场景文件的 `style.voice.persona`，语速取 `style.voice.speed`（atempo 变速，不变调）。

### 句级缓存

每句的原声缓存在音频目录的 `.cache/` 里：`.cache/tts/<键>.json` 是索引，指向 `.cache/objects/<内容的 sha256>.wav`。键是 sha256(缓存格式版本、模型、声音、人设、语气、语言、送给 TTS 的 instructions、送给 TTS 的正文——也就是 normalize、去掉书签之后的读法文本)。所以改了某一句，只有这一句重新调用 TTS；同一段的其他句子直接读盘，不花钱、不等待，韵律也不变（拼出来的 `<beat>.wav` 里这几句的采样逐字节相同）。日志每段一行，例如 `Hook: 2 cached, 1 synthesized；3 句，9.4s`。

- 语速不在原声的键里：`speed` 是本地的 atempo，变速后的音频另存一个条目（键是原声字节的 sha256 + speed 的完整精度 repr）。改语速会重拼，但不会重新调用 TTS。
- 句子音频和停顿（`silence_after` / `hold`）都和上次拼的时候一样，就不重拼 `<beat>.wav`。align 写进 timing.json 的书签时刻只在句子 key 和书签位置也没变时保留；只挪了书签、改了书签名或句子 id 时，音频照用，但过期的 `marks` / `check` / `words` 会被清掉，要重新 align。`--force` 照样重拼，句子仍读缓存；`--no-cache` 每句都重新调用 TTS 并刷新缓存。
- 写音频目录的命令整个运行都独占它（`.cache/synth.lock`，flock，跨进程）：`synth`（读写缓存、发布 wav / timing / 说明、写 manifest 和时间报告）、`align`（把书签写回 timing.json）、`pitch --fix`。同一个目录上的第二个 synth 默认打一行 `waiting for another synth on <目录>` 等前一个跑完；加 `--no-wait` 则立刻报错退出（退出码 2）。只读的 `review` 和不带 `--fix` 的 `pitch` 不加锁。
- 发布一段的顺序：先原子地把 manifest 里这一段记成 `pending`（来源不明、不合规）→ 预写 `.cache/beats-next/<beat>.json`（这次 wav 的 sha256 + model / voice / speed）→ 换 `<beat>.wav` → 写 timing.json → 写 `.cache/beats/<beat>.json`（同样带 sha256、cues 和来源）→ 把实际来源提交进 manifest。来源也算在“没变就沿用”的指纹里。每次 synth（和 `pitch --fix`）开头先按说明核对 manifest：哪份说明的 sha256 和现在的 wav 对得上，就按它的来源改正记录；都对不上、记录又是 pending / 来源不明，就记成来源不明。所以在任何一步被打断，manifest 都不会把不合规的音频说成合规，下一次运行（哪怕只合成别的 beat）会把记录修好。没有任何说明的旧目录保留原来的记录。manifest、时间报告都是原子替换。
- 命中要求索引和格式版本对得上、对象的 sha256 就是它的文件名、wav 头合法且帧数一致；不满足（写到一半被打断、文件被截断或改过）就当作损坏，提示后重念。
- 音频对象不可变：同名的文件只会是同样的内容，已经有完好的就不再写。`--no-cache`、`pitch --fix` 刷新同一个键时只写一个新对象、原子地换索引，已经拿到旧对象的读者读到的仍是完整的旧音频。变速条目的键是原声字节的 sha256，算键的字节和喂给 ffmpeg 的是同一份。
- `lines/<beat>-<下标>.wav`（+ `.json`）是给 `align` 复核和 `pitch` 用的副本，每次按缓存重写，不再当缓存读。升级前合成的目录：副本的说明（文本、instructions、声音、模型）完全对得上就收进缓存，不会重新花钱。`pitch --fix` 仍可沿用人设文案改过之前的副本（只核对文本、声音、模型）。
- 旧对象不会自动删除，删掉 `.cache/` 就是清空缓存；它不进 manifest，也不影响 kit。

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

要人看的提示都带场景文件里的位置，`t` 是这句在 beat 音频里的起止：

```
⚠ beats[2].say[1] (line 37, t=12.300-15.800): 书签 {@d1} 插值（interpolated）在 13.420s，请核对
⚠ beats[2].say[1] (line 37, t=12.300-15.800): 匹配 80%；没听到 ['加回']；多出 []；复核仍缺 ['加回']
```

行号指向这句的 `text`（`--lang en` 时是 `en`），从 `yaml.compose` 的节点起点取，scene-spec 不用改。

## 审片

`review` 把一集成片交给 Gemini 的视频理解（默认 `gemini-4-argon`，2026-10 时面向数学和长视频的旗舰模型；`--model` 或环境变量 `GEMINI_REVIEW_MODEL` 可换）。模型要同时看画面和听声音，对照场景稿里的旁白、书签时刻和 align 写下的回听差异（有 `--spec` / `--audio` 才附上）。提示词在 `src/voice/prompts/review/v1/`，换说法就加一个新版本目录，用 `--prompt-version` 指定。

凭证和 TTS 一样，只从 `--env` 或环境变量读，不写进仓库，也不打印：

- `GEMINI_API_KEY`：开发者 API。长视频走 Files API 上传。有这把钥匙就优先用它。
- 没有的话，用 Vertex，和 `synth` 同一套：`GOOGLE_CLOUD_PROJECT`、`GOOGLE_API_KEY`，区域默认 `GOOGLE_CLOUD_REGION`（没有则 `us-central1`）。
- `ANIMIND_REVIEW_BACKEND=gemini|vertex` 可以强制选一条。

名字写成 OpenRouter 那种 `google/gemini-4-argon` 也可以，审片会去掉 `google/` 再走 google.genai。审片不走 OpenRouter 的语音接口。

短于 `--max-chunk-seconds`（默认 300 秒）的片子一次看完，不重新编码。更长的片子用 ffmpeg 切成重叠窗口（默认重叠 8 秒），每段单独看，时间戳加回整集，重叠处同一类问题会合并。单段不超过 15MB 时内联上传，更大就走 Files API。数学画面默认 `--media-resolution high`，方便读公式。`--media-processing` 默认 `static`（大约每秒一帧，布局问题不容易漏）；`agentic` 更省 token，模型会自己挑片段看，可能放过重叠和出框。

每次 HTTP 调用追加一行到报告旁边的 `review_usage.jsonl`（和 `tts_usage.jsonl` 一样可累计）。这一次报告里的费用只累加这次内存里的调用，日志每行带 `run_id`，不会去读共享日志的最后几行，所以几集同时审不会把别人的费用算进来。API 返回了 token 数就按那个算钱；流在中途断了就保留已经收到的用量。一个 token 都没拿到（或拿到的是 0）就按文字加视频时长粗估，并在记录里标 `token_source: estimate`，不会记成已经确认的 0。价目是标准档，写在 `src/voice/review.py` 的 `PRICES`（2026-10-20 对照 [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing)）：

| 模型 | 输入 / 输出（美元 / 百万 token） | 何时涨价 |
| --- | --- | --- |
| `gemini-4-argon`（默认） | 2 / 10 | 2026-12-01 起 4 / 20 |
| `gemini-3.1-pro-preview` | 2 / 12；prompt 超过 20 万 token 时 4 / 18 | — |
| `gemini-3.8-flash` | 0.75 / 3.75 | 2027-01-01 起 1.50 / 7.50 |

不在表里的模型按 `gemini-4-argon` 估价，记录里 `price_known` 为 false。高分辨率静态视频大约 300 token/秒（含音频），低分辨率大约 100 token/秒，所以一段 5 分钟、一次看完、高分辨率、输出大约 1 千 token 的请求，在 2026-11-30 之前大约是 `90_000 × $2/1M + 1_000 × $10/1M ≈ $0.19`。15 分钟切成三段大约三倍。这是估价；以 `review_usage.jsonl` 里的 `est_usd` 为准。思考 token 算在输出里。

报告写两份：`--out` 指向的 JSON，以及把后缀换成 `.md` 的短摘要。`--out` 如果是目录，文件名是 `<视频名>.review.json`。退出码和以后做门槛时一样：0 通过，1 未通过，2 是用法、凭证或模型一直返回坏 JSON。坏 JSON 会把校验错误塞回提示词再要一次，最多 3 次；429 和 5xx 按指数退避重试，次数由 `--attempts` 定。

默认门槛：任何 `major` 或 `blocker` 都不通过，`minor` 和 `info` 只记下来。用 `--thresholds` 换一套 JSON：

```json
{"fail_severities": ["blocker", "major"], "max_counts": {"minor": 8}, "fail_categories": {"math": "minor"}}
```

`fail_categories` 表示这类问题从哪个级别起算失败（上面是：数学问题只要 minor 也失败）。`max_counts` 是某一级条数的上限，超过会多一条 `RV001`。

JSON 里给流水线看的是 `gate`，形状和 scene-spec 检查器的 `Report.to_dict()` 一样：`ok`、`issues`（`level` / `code` / `path` / `message`）、`stats`。`level` 为 `error` 的条目才让 `ok` 变成 false；`path` 是问题在场景文件里的位置（见下面「追溯到场景文件」），追溯不到时是整集时间轴上的 `t=<start>-<end>`。`gate` 不是 scene-spec `check_timing` 的输入。`check_timing` 要的是 `{beats: ...}`（旁白和书签时刻），不要把 `gate` 或整份审片 JSON 交进去。稳定代码：

| 代码 | category | 在看什么 |
| --- | --- | --- |
| RV100 | layout_overlap | 文字或图形重叠 |
| RV110 | out_of_frame | 元素出框 |
| RV120 | legibility | 字太小 |
| RV200 | av_sync | 画面和旁白不同步（含 CUE LATE） |
| RV210 | tts_pronunciation | 吞字、错读、同音字（背/被） |
| RV220 | tts_filler | 稿子里没有的诶/嗯/哎 |
| RV230 | tts_voice_drift | 听起来换了一个人 |
| RV300 | pedagogy | 跳步、画面和讲解打架 |
| RV310 | math | 式子或数字是错的 |
| RV400 | seam | 像两集硬接在一起 |
| RV900 | other | 其他会退回的问题 |

顶层的 `ok` 和 `gate.ok` 相同。完整字段见 `src/voice/prompts/review/v1/report.schema.json`。

### 追溯到场景文件

有 `--spec` 时，每条问题按它在整集上的时间 `[start, end]` 对回场景文件，补上 `beat`、`sentence`（句子的 id，没有 id 就是下标）、`line`、`column` 和 `path`，例如 `beats[3].say[1] (line 22, t=15.200-15.400)`：

- 每个 beat 在成片里的起止取 kit 写在成片旁边的 `<成片>.chapters.json`（或 `--chapters` 指定）；句子在 beat 里的时刻取 `--audio` 的 timing.json，再加上它之前的子步定格（chapters 的 `steps`，成片里定住、旁白插了静音）。
- 问题和哪句旁白的时间窗重叠最多就对到哪句；落在两句之间、`pause` 的停顿里、没有旁白的 beat 里，只对到 beat（`line` 是这个 beat 的 `id` 那一行，`path` 是 `beats[i] (...)`）；不在任何 beat 里就不动。
- 模型自己填了 `beat` / `sentence` 的一律保留，`line` / `path` 指向模型说的位置；和时间轴对不上（或者场景里没有这个 beat / 句子）就写进 `trace_note`。
- 没有 chapters.json 时，只有每个 beat 都有 timing.json 才按旁白长度估（kit 的下限：旁白 + 0.28 秒 + `pause.hold`），动画比旁白长时会偏早；`sources.timeline` 写明用的是 `chapters` 还是 `estimated`。都没有就不追溯。

匹配率低于 85%，或者漏念 / 多念两个字以上的句子会被标出来，例如 §6.7 第 01 支里：

```
⚠ Problem[s4] 匹配 95%；没听到 ['后面']；多出 []
  听到：然后告诉我们，数列 A n 是二、五、十一、二十一、三十六，省略，而且，……
```

## 开发

```bash
uv run pytest      # 不调用 TTS、Whisper 和 Gemini：合成用假的 speak，对齐用假的 transcriber，审片用录好的 JSON
ruff check .
```

## 作者与许可

作者：Boxi Yu（[@BoxiYu](https://github.com/BoxiYu)）

代码以 [MIT 许可](LICENSE)发布。
