# 01 - Sparse Autoencoders

本目录包含三个 SAE 训练笔记本，覆盖不同的模型架构和激活提取方案。

## Notebooks

### 1. [training_a_sparse_autoencoder.ipynb](training_a_sparse_autoencoder.ipynb)

使用 **SAELens** 在 decoder-only 模型 (tiny-stories-1L-21M) 上训练 SAE 的入门教程。

- 模型加载：`HookedTransformer.from_pretrained("tiny-stories-1L-21M")`
- 激活提取：SAELens 内置的 `SAETrainingRunner` + `ActivationsStore`
- 目标层：`blocks.0.hook_mlp_out`
- 无需手动实现 SAE，开箱即用

### 2. [training_sae_t5_transformerlens.ipynb](training_sae_t5_transformerlens.ipynb)

使用 **TransformerLens** 的 `HookedEncoderDecoder` 提取 T5-large 内部激活，手动训练 SAE。

- 模型加载：`HookedEncoderDecoder.from_pretrained("google-t5/t5-large")`
- 激活提取：`model.run_with_cache(text)` -> `cache["encoder.12.hook_mlp_out"]`
- 输入格式：直接传字符串
- 优化：`names_filter` 只缓存目标 hook，节省显存

### 3. [training_sae_t5_nnsight.ipynb](training_sae_t5_nnsight.ipynb)

使用 **nnsight** 提取 T5-large 内部激活，手动训练 SAE。

- 模型加载：`LanguageModel("google-t5/t5-large", automodel=AutoModelForSeq2SeqLM)`
- 激活提取：`model.trace(input_ids)` -> `model.encoder.block[12].layer[1].output.save()`
- 输入格式：需要先 tokenize 为 token IDs
- 优势：支持更细粒度的干预操作

## TransformerLens vs nnsight 对比

| | TransformerLens | nnsight |
|---|---|---|
| 模型加载 | `HookedEncoderDecoder.from_pretrained()` | `LanguageModel(automodel=AutoModelForSeq2SeqLM)` |
| 激活提取 | `run_with_cache` 一次返回全部 cache | `trace` 上下文 + `.save()` |
| 输入格式 | 字符串 | Token IDs |
| 记忆优化 | `names_filter` 按需缓存 | 逐个 `.save()` |
| Logit lens | `model.W_E` 直接获取 | `t5_model.decoder.embed_tokens.weight` |

两个版本的 SAE 实现（类、训练循环、评估）完全一致，仅激活提取方式不同。

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
