# BatchTopK SAE 训练详解

基于 `03_training_batchtopk_sae_t5_decoder.ipynb`，面向 SAE 初学者的逐步讲解。

## 0. SAE 是什么？为什么要用它？

神经网络某一层的输出是一个稠密向量（比如 1024 维），每个维度可能同时编码多种信息（多义性）。Sparse Autoencoder 的目标是：

```
输入: [1024维稠密向量]
        ↓ encode (W_enc)
隐藏层: [16384维稀疏向量] ← 只有 ~100 个非零
        ↓ decode (W_dec)
输出: [1024维重建向量] ≈ 输入
```

扩展维度 + 稀疏约束 → 每个 feature 更容易对应一个可解释的概念（单义性）。

## 1. 对谁训练 SAE？

模型：**T5-large**（encoder-decoder 架构），训练位置：**decoder 第 12 层 MLP 输出**。

```
源文本 → [Encoder 24层] → 编码表示
                              ↓ cross-attention
目标前缀 → [Decoder 第1层] → ... → [第12层 MLP输出] ← SAE 在这里
                              → ... → [第24层] → 输出
```

decoder 激活包含 cross-attention 到 encoder 的信息，理解它有助于理解模型怎么生成输出。

## 2. 怎么拿到 decoder 的激活值？

T5 decoder 需要两样东西：encoder 输入（源文本）和 decoder 输入（目标文本）。

```python
logits, cache = model.run_with_cache(
    enc_tokens.input_ids,              # 源文本
    decoder_input=dec_tokens.input_ids, # 目标文本
    names_filter=lambda name: name == "decoder.12.hook_mlp_out",
)
acts = cache["decoder.12.hook_mlp_out"]  # [1, seq_len, 1024]
```

如果只给 encoder 输入不给 decoder 输入，decoder 无法做 cross-attention，激活值无意义或直接报错。

## 3. 数据集：XSum

用 XSum 摘要数据集，提供 encoder-decoder 对：
- Document（文章）→ encoder 输入
- Summary（摘要）→ decoder 输入

`streaming=True` 逐条读取，不一次性下载整个数据集。

## 4. 核心超参数

| 参数 | 值 | 含义 |
|---|---|---|
| `d_in` | 1024 | T5 d_model，SAE 输入维度 |
| `d_sae` | 16384 | SAE 隐藏层维度（16x 扩展） |
| `k` | 100.0 | 每个样本平均激活的 feature 数 |
| `batch_size` | 4096 | 每步训练的 token 数 |
| `total_steps` | 10000 | 总训练步数 |

`k=100` 意味着 16384 个 feature 中只有 ~0.6% 是激活的，非常稀疏。

## 5. batch 和 seq_len 是什么？

**seq_len**：一句话有多少个 token。

```
"AI is amazing" → tokenize → [234, 16, 3456]  → seq_len = 3
```

每个 token 产生一个 1024 维向量，所以一个句子的激活值 shape 是 `[seq_len, 1024]`。

**batch**：同时处理几句话。GPU 擅长并行计算，把多句话打包处理效率更高。

**在本 notebook 中**：每次只处理 1 条样本（batch=1），然后把多条样本的激活拼起来凑够 `batch_size=4096` 个 token：

```
样本1 摘要: 5 个 token → [5, 1024]
样本2 摘要: 8 个 token → [8, 1024]
样本3 摘要: 3 个 token → [3, 1024]
            ↓ torch.cat
    拼起来: [16, 1024]  ← 继续读取直到凑够 4096 个 token
```

## 6. 为什么要归一化激活值？

不同层、不同模型的激活值大小差异很大。SAE 的权重是随机初始化的（值在 0 附近），如果输入激活值很大，`x @ W_enc` 会产生很大的值，梯度也会很大，训练不稳定。

解决方案：缩放到标准范围，让每个 token 的 L2 范数 ≈ `sqrt(d_in)` = `sqrt(1024)` ≈ 32。

```python
# 估计 50 个 batch 的平均范数
mean_norm = np.mean(norms)  # 比如 200
scaling_factor = sqrt(1024) / mean_norm  # 32 / 200 = 0.16

# 训练时应用
sae_in = raw_activations * scaling_factor  # 缩放后范数 ≈ 32
```

## 7. BatchTopK SAE 结构

```
输入 x: [batch, 1024]
    ↓
x_normalized = x × scaling_factor
    ↓
pre_act = x_normalized @ W_enc + b_enc    → [batch, 16384]
    ↓
top-k 选择：只保留激活值最大的 k 个，其余置零
    ↓
feature_acts: [batch, 16384]（稀疏的）
    ↓
reconstruction = feature_acts @ W_dec + b_dec  → [batch, 1024]
```

## 8. Standard SAE vs BatchTopK SAE

| | Standard SAE | BatchTopK SAE |
|---|---|---|
| 稀疏方式 | ReLU + L1 惩罚 | 全局 top-k 选择 |
| 稀疏控制 | L1 coefficient | k（激活数） |
| Loss | MSE + L1 | MSE + aux loss |
| 死神经元处理 | L1 warm-up | aux loss 强制激活 |
| 直观程度 | 间接（调系数看效果） | 直接（只保留 k 个） |

## 9. 训练循环拆解

```python
for step in range(10000):
    # 1. 收集激活
    sae_in = collect_activations_batch(...)  # [4096, 1024]

    # 2. 归一化
    sae_in = sae_in * sae.scaling_factor

    # 3. 前向传播（loss 由 SAELens 内部计算）
    step_output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=sae_in,
            coefficients={},  # BatchTopK 不需要 L1！
            ...
        )
    )

    # 4. 反向传播 + 梯度裁剪 + 更新
    step_output.loss.backward()
    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
    optimizer.step()
```

**Loss = MSE + Aux Loss**：
- MSE Loss：重建值和原始激活的均方误差，衡量信息保留程度
- Aux Loss：辅助损失，专门救活死神经元

`coefficients={}` 是空的，因为 BatchTopK 的稀疏靠 top-k 选择直接控制，不需要 L1 coefficient。

## 10. 评估指标

| 指标 | 含义 | 好的表现 |
|---|---|---|
| Explained Variance | SAE 解释了多少原始方差 | > 0.8 |
| L0 | 每个 token 平均激活几个 feature | 接近 k=100 |
| Dead features | 从不激活的 feature 数量 | 越少越好 |
| Feature density | 每个 feature 的激活频率 | 集中在低值 |

## 11. Logit Lens

把 SAE 的 decoder 权重投影到 T5 的词表空间，看每个 feature 喜欢哪些 token：

```python
embed = model.W_E                    # T5 embedding 矩阵
projection = sae.W_dec @ embed.T     # [16384, vocab_size]
```

```
Feature  491: ['sixty', 'seventeen', 'twenty', 'fifty']  ← 数字 feature
Feature  604: ['thinking', 'spending', 'switching']       ← 动名词 feature
```

## 12. batch_size 怎么选？

batch_size 是超参数，受 GPU 显存限制。越大训练越稳定，但超过某点收益递减。

| GPU 显存 | 建议 batch_size |
|---|---|
| 8 GB | 512–1024 |
| 16 GB | 2048–4096 |
| 24 GB | 4096–8192 |
| 80 GB (A800) | 16384–32768 |

A800 有 80GB 显存，T5-large (~3GB) + SAE 训练 (~12GB at batch_size=32768) 总共不到 20GB，非常充裕。

建议：先用小 batch (512) 试通代码，再改大正式训练。

## 整体流程总结

```
1. 加载 T5-large
2. 从 XSum 收集 decoder.12 的激活值
3. 估计 scaling factor（归一化激活）
4. 训练 BatchTopK SAE（10000 步）
5. 评估：EV、L0、dead features
6. Logit Lens：看每个 feature 代表什么
7. 保存模型
```
