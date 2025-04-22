import os
import time
from mybpe import BpeTokenizer
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置镜像源

with open('data/manual.txt', 'r', encoding="utf-8") as f:
    text = f.read()

tokenizer = BpeTokenizer()
tokenizer.train(text, vocab_size=1024, verbose=True)
tokenizer.save("my_bpe_tokenizer")  # 保存到my_bpe_tokenizer.model和my_bpe_tokenizer.vocab

loaded_tokenizer = BpeTokenizer.load("my_bpe_tokenizer")
encoded = loaded_tokenizer.encode(text)
decoded = loaded_tokenizer.decode(encoded)
print("bpe:",decoded)  # "hello world"

with open('result/manual.txt', 'w', encoding='utf-8') as f:
    f.write(decoded)

from transformers import GPT2Tokenizer

# 加载预训练的 GPT-2 Tokenizer
tokenizerGpt2 = GPT2Tokenizer.from_pretrained("gpt2")  # 使用 "gpt2-medium" 等更大模型需更改名称
text_t1 = "Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university."
text_t2 = "博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。"
# test1
encoded_t1_gpt = tokenizerGpt2.encode(text_t1)
encoded_t1_bpe = loaded_tokenizer.encode(text_t1)
print("gpt2 tokrnizer: length= ",len(encoded_t1_gpt))
print(encoded_t1_gpt) # 解码为字符串
print("my bpe tokrnizer: length= ",len(encoded_t1_bpe)) # 解码为字符串
print(encoded_t1_bpe)

# test2
encoded_t2_gpt = tokenizerGpt2.encode(text_t2)
encoded_t2_bpe = loaded_tokenizer.encode(text_t2)
print("gpt2 tokrnizer: length= ",len(encoded_t2_gpt))
print(encoded_t2_gpt) # 解码为字符串

print("my bpe tokrnizer: length= ",len(encoded_t2_bpe)) # 解码为字符串
print(encoded_t2_bpe)
#print("gpt2 tokrnizer: ",tokenizerGpt2.decode(encoded_t2_gpt)) # 解码为字符串
print("my bpe tokrnizer: ",loaded_tokenizer.decode(encoded_t2_bpe)) # 解码为字符串
