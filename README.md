# animind-voice

[scene-spec](https://github.com/animind-ai/scene-spec) 的旁白流水线：读场景文件里的 `say`，逐句合成旁白，再用 Whisper 求出每个书签 `{@名字}` 的真实时刻，顺带检查 TTS 有没有漏念、多念。输出直接给 [kit](https://github.com/animind-ai/kit) 渲染用。

## 两个命令

```bash
uv sync --extra align
# 1. 合成：每句一段音频（按文本 + 语气 + 人设 + 模型缓存），拼成 <beat>.wav + <beat>.timing.json
uv run animind-voice synth ../scene-spec/examples/s67-p01.scene.yaml --out build/audio --env /path/to/.env
# 2. 对齐 + 回听：把书签的真实时刻写回 timing.json，列出需要人去听的句子
uv run animind-voice align ../scene-spec/examples/s67-p01.scene.yaml --audio build/audio
```

`synth` 的模型和声音从 `--env` 文件或环境变量读：`GEMINI_TTS_MODEL`（名字带 `/` 走 OpenRouter，需要 `OPENROUTER_API_KEY`；否则走 Vertex，需要 `uv sync --extra vertex` 和 `GOOGLE_*`）、`GEMINI_TTS_VOICE`。密钥从不打印，也不要放进仓库。人设取场景文件的 `style.voice.persona`，语速取 `style.voice.speed`（atempo 变速，不变调）。

合成之前，每句旁白先交给 [zh-math-reading](https://github.com/animind-ai/zh-math-reading)：`say` 里的 `$...$` 公式展开成读法（`$a_{n+1}^2$` → “a n 加一的平方”），书签位置按展开后的文字计算；同时打印读法风险提示（冒号后紧跟字母、长句、多音字），只提示、不拦截。不含公式的句子读法逐字节不变，已有的音频缓存继续有效。zh-math-reading 是同级目录的可编辑路径依赖，和 scene-spec 一样要并排检出。

`align` 在本机用 [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（Apple 芯片）跑 `whisper-large-v3-turbo`，不需要联网 API。做法：

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

## 开发

```bash
uv run pytest      # 不调用 TTS 和 Whisper：合成用假的 speak，对齐用假的 transcriber
ruff check .
```

## 作者与许可

作者：Boxi Yu（[@BoxiYu](https://github.com/BoxiYu)）

代码以 [MIT 许可](LICENSE)发布。
