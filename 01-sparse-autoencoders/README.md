# 01 - Sparse Autoencoders

本目录包含四个 SAE 训练笔记本，覆盖不同的模型架构、激活提取方案和目标层。

## Notebooks

### 1. [training_a_sparse_autoencoder.ipynb](training_a_sparse_autoencoder.ipynb)

使用 **SAELens** 在 decoder-only 模型 (tiny-stories-1L-21M) 上训练 SAE 的入门教程。

- 模型加载：`HookedTransformer.from_pretrained("tiny-stories-1L-21M")`
- 激活提取：SAELens 内置的 `SAETrainingRunner` + `ActivationsStore`
- 目标层：`blocks.0.hook_mlp_out`
- 无需手动实现 SAE，开箱即用

### 2. [training_sae_t5_transformerlens.ipynb](training_sae_t5_transformerlens.ipynb)

使用 **TransformerLens** 提取 T5-large **encoder** 激活，手动训练 SAE。

- 模型加载：`HookedEncoderDecoder.from_pretrained("google-t5/t5-large")`
- 激活提取：`model.run_with_cache(text)` -> `cache["encoder.12.hook_mlp_out"]`
- 数据集：C4（纯文本，只需 encoder 输入）
- 优化：`names_filter` 只缓存目标 hook，节省显存

### 3. [training_sae_t5_nnsight.ipynb](training_sae_t5_nnsight.ipynb)

使用 **nnsight** 提取 T5-large **encoder** 激活，手动训练 SAE。

- 模型加载：`LanguageModel("google-t5/t5-large", automodel=AutoModelForSeq2SeqLM)`
- 激活提取：`model.trace(input_ids)` -> `model.encoder.block[12].layer[1].output.save()`
- 数据集：C4
- 优势：支持更细粒度的干预操作

### 4. [training_sae_t5_decoder_transformerlens.ipynb](training_sae_t5_decoder_transformerlens.ipynb)

使用 **TransformerLens** 提取 T5-large **decoder** 激活，手动训练 SAE。

- 模型加载：同上，`HookedEncoderDecoder`
- 激活提取：`model.run_with_cache(enc_ids, decoder_input=dec_ids)` -> `cache["decoder.12.hook_mlp_out"]`
- 数据集：XSum（摘要任务，需要 encoder + decoder 输入对）
- 特点：decoder 激活包含 cross-attention 信息，与 encoder 激活本质不同

### 5. [training_sae_t5_demo.ipynb]
改动清单

| 项目              | 原版                            | Demo 版                      |
| --------------- | ----------------------------- | --------------------------- |
| 模型              | t5-large (d_model=1024, 24 层) | t5-small (d_model=512, 6 层) |
| Hook 点位         | decoder.12.hook_mlp_out       | decoder.3.hook_mlp_out      |
| SAE 扩展维度        | d_sae=16384 (16 倍扩展)          | d_sae=1024 (2 倍扩展)          |
| 总训练步数           | 50000                         | 200                         |
| 批次大小 Batch size | 4096                          | 512                         |
| 学习率             | 1e-4                          | 1e-3                        |
| L1 预热步数         | 2500 步                        | 10 步                        |
| 上下文 / 目标序列长度    | 128 / 64                      | 64 / 32                     |
| 归一化估计批次数量       | 100                           | 5                           |
| 日志打印频率          | 每 100 步                       | 每 10 步                      |
| Wandb 日志        | 完整集成                          | 直接移除                        |
| 评估可视化图表         | 6 张子图 + 完整详细分析                | 3 张子图（loss/EV/L0）           |
| 代码单元格数量         | 约 29 个                        | 17 个                        |
| 导入依赖            | 分散多单元格                        | 统一合并至单个单元格                  |
## TransformerLens vs nnsight 对比

| | TransformerLens | nnsight |
|---|---|---|
| 模型加载 | `HookedEncoderDecoder.from_pretrained()` | `LanguageModel(automodel=AutoModelForSeq2SeqLM)` |
| 激活提取 | `run_with_cache` 一次返回全部 cache | `trace` 上下文 + `.save()` |
| 输入格式 | 字符串 | Token IDs |
| 记忆优化 | `names_filter` 按需缓存 | 逐个 `.save()` |
| Logit lens | `model.W_E` 直接获取 | `t5_model.decoder.embed_tokens.weight` |
| Decoder 支持 | `run_with_cache(enc, decoder_input=dec)` | `model.trace(enc_ids)` + decoder proxy |

## Encoder vs Decoder SAE 对比

| | Encoder SAE | Decoder SAE |
|---|---|---|
| Hook 点 | `encoder.N.hook_mlp_out` | `decoder.N.hook_mlp_out` |
| 输入 | 只需源文本 | 需要源文本 + 目标文本 |
| 数据集 | C4（任意文本） | XSum（摘要任务） |
| 激活特点 | 纯编码表示 | 包含 cross-attention 到 encoder 的信息 |
| 应用场景 | 理解模型如何编码输入 | 理解模型如何生成输出 |

## 通用超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `d_in` | 1024 | T5-large d_model |
| `d_sae` | 16384 | 16x expansion |
| `l1_coefficient` | 5.0 | 稀疏度控制 |
| `l1_warm_up_steps` | 2500 | 总步数的 5% |
| `decoder_init_norm` | 0.1 | W_dec 行归一化 |
| `lr` | 1e-4 | Adam 学习率 |
| `batch_size` | 4096 | 每步 token 数 |
| `total_steps` | 50000 | 总训练步数 |

## 为什么需要 TransformerLens / nnsight？

SAELens 不支持 T5 等 encoder-decoder 模型 — 它的 `load_model` 只处理 `HookedTransformer`、`HookedMamba` 和 `AutoModelForCausalLM`。T5 必须通过 `HookedEncoderDecoder`（TransformerLens）或 `LanguageModel` + `AutoModelForSeq2SeqLM`（nnsight）加载。

## References

- [SAELens](https://github.com/decoderesearch/SAELens)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [nnsight](https://nnsight.net/)
- [Scaling Monosemanticity](https://www.anthropic.com/research/mapping-mind-language-model)
