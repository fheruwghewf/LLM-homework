from Tokenizer import Tokenizer

file_path = "manual.txt"
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# print(content)

t = Tokenizer()

t.train(content, 1024)

print(t.decode(t.encode(content)) == content)