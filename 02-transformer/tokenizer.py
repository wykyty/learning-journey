from transformers import AutoTokenizer

model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "春眠不觉晓"
# 编码：文本 -> Token ID
encoded = tokenizer(text)
print("Token ID:", encoded["input_ids"])

# 看每个 ID 对应什么片段
tokens = tokenizer.tokenize(text)
print("Token 片段:", tokens)

# 解码：Token ID -> 文本
decode = tokenizer.decode(encoded["input_ids"])
print("还原文本：", decode)

"""
Token ID: [101, 3217, 4697, 679, 6230, 3236, 102]
Token 片段: ['春', '眠', '不', '觉', '晓']
还原文本： [CLS] 春 眠 不 觉 晓 [SEP]
"""