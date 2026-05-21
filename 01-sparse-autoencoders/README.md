# 01 - Sparse Autoencoders

在 T5 等模型上训练 Sparse Autoencoder (SAE) 的学习笔记本，覆盖不同的 SAE 类型、训练方式和目标层。

## Notebooks

### [01_training_a_sparse_autoencoder.ipynb](01_training_a_sparse_autoencoder.ipynb)

SAELens 入门教程，在 decoder-only 模型上训练 SAE，开箱即用。来源：https://github.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb

- 模型：`tiny-stories-1L-21M` (HookedTransformer)
- 目标层：`blocks.0.hook_mlp_out`
- 方式：`SAETrainingRunner` 一键训练，无需手动实现

### [02_training_sae_t5_decoder.ipynb](02_training_sae_t5_decoder.ipynb)

用 SAELens 的 `StandardTrainingSAE` 训练 T5-large **decoder** 侧 SAE。

- 模型：`google-t5/t5-large` (HookedEncoderDecoder)
- 目标层：`decoder.12.hook_mlp_out`
- 数据集：XSum（摘要任务，提供 encoder-decoder 对）
- 方式：手动训练循环 + SAELens 的 `training_forward_pass`

### [03_training_batchtopk_sae_t5_decoder.ipynb](03_training_batchtopk_sae_t5_decoder.ipynb)

用 SAELens 的 `BatchTopKTrainingSAE` 训练 T5-large **decoder** 侧 SAE。

- 模型/数据/目标层同 Notebook 02
- 稀疏控制：`k`（batch 平均激活数）替代 L1 coefficient
- Loss：MSE + dead neuron aux loss（无 L1）
- 推理时转换为 JumpReLU

### [04_training_sae_t5_decoder_transformerlens.ipynb](04_training_sae_t5_decoder_transformerlens.ipynb)

纯 TransformerLens + PyTorch 手动实现 SAE，不依赖 SAELens。

- 模型：`google-t5/t5-large` (HookedEncoderDecoder)
- 目标层：`decoder.12.hook_mlp_out`
- 数据集：XSum
- SAE 架构、loss、训练循环全部从零实现

### [05_training_sae_t5_demo.ipynb](05_training_sae_t5_demo.ipynb)

轻量级 demo，用于快速验证训练流水线。

- 模型：`google-t5/t5-small`（d_model=512, 6 层）
- 目标层：`decoder.3.hook_mlp_out`
- SAE：d_sae=1024（2x 扩展）
- 训练：200 步，batch_size=512

## Notebook 关系

```text
01  SAELens 入门 (decoder-only, tiny-stories)
 │
 ├── 02  Standard SAE on T5 decoder (SAELens 类 + 手动循环)
 ├── 03  BatchTopK SAE on T5 decoder (SAELens 类 + 手动循环)
 ├── 04  手写 SAE on T5 decoder (纯 PyTorch)
 └── 05  Demo (t5-small, 快速验证)
```

## Standard SAE vs BatchTopK SAE

| | StandardTrainingSAE | BatchTopKTrainingSAE |
|---|---|---|
| 稀疏控制 | L1 coefficient | k (float) |
| Loss | MSE + L1 | MSE + aux loss |
| L1 warm-up | 有 | 无 |
| 额外指标 | — | topk_threshold (EMA) |
| 推理格式 | ReLU | JumpReLU |

## 为什么用 TransformerLens 而不用 SAELens？

SAELens 的 `SAETrainingRunner` 不支持 `HookedEncoderDecoder`（T5 等 encoder-decoder 模型），只能处理 `HookedTransformer`、`HookedMamba` 和 `AutoModelForCausalLM`。因此需要通过 TransformerLens 加载 T5，手动提取激活后训练 SAE。

## 通用超参数（T5-large decoder SAE）

| 参数 | 值 | 说明 |
|---|---|---|
| `d_in` | 1024 | T5-large d_model |
| `d_sae` | 16384 | 16x 扩展 |
| `l1_coefficient` | 5.0 | 稀疏度控制 (Standard SAE) |
| `k` | 100.0 | 平均激活数 (BatchTopK) |
| `lr` | 1e-4 | Adam 学习率 |
| `batch_size` | 4096 | 每步 token 数 |
| `total_steps` | 10000–50000 | 总训练步数 |

## References

- [SAELens](https://github.com/decoderesearch/SAELens)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- [Scaling Monosemanticity](https://www.anthropic.com/research/mapping-mind-language-model)
