# My tokenizer
from Tokenizer import Tokenizer as MyTokenier

file_path = "manual.txt"
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# print(content)

t = MyTokenier()

t.train(content, 1024)

print(t.decode(t.encode(content)) == content)

test_content = 'Originated as the Imperial University of Peking in 1898, Peking University was China’s first national comprehensive university and the supreme education authority at the time. Since the founding of the People’s Republic of China in 1949, it has developed into a comprehensive university with fundamental education and research in both humanities and science. The reform and opening-up of China in 1978 has ushered in a new era for the University unseen in history. And its merger with Beijing Medical University in 2000 has geared itself up for all-round and vibrant growth in such fields as science, engineering, medicine, agriculture, humanities and social sciences. Supported by the “211 Project” and the “985 Project”, the University has made remarkable achievements, such as optimizing disciplines, cultivating talents, recruiting high-caliber teachers, as well as teaching and scientific research, which paves the way for a world-class university.'
test_content2 = '博士学位论文应当表明作者具有独立从事科学研究工作的能力，并在科学或专门技术上做出创造性的成果。博士学位论文或摘要，应当在答辩前三个月印送有关单位，并经同行评议。学位授予单位应当聘请两位与论文有关学科的专家评阅论文，其中一位应当是外单位的专家。评阅人应当对论文写详细的学术评语，供论文答辩委员会参考。'
# tokenizer from hugging face
from tokenizers import Tokenizer
tokenizer: Tokenizer = Tokenizer.from_pretrained("gpt2")

print('test1:')
print('my encoder: len = ', len(t.encode(test_content)))

print('gpt2 encoer: len = ', len(tokenizer.encode(test_content)))

print('test2')
print('my encoder: len = ', len(t.encode(test_content2)))

print('gpt2 encoer: len = ', len(tokenizer.encode(test_content2)))
