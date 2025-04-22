import unicodedata
import os
import time
import regex as re
import json
import pickle

def get_stats(ids , counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        # if pair in counts:
        #     counts[pair]+=1
        # else:
        #     counts[pair]+=0
        #从字典 counts 中查找键 pair 对应的值, pair存在，返回对应的值,pair 不存在，返回0
        counts[pair] = counts.get(pair,0)+1
    return counts
        
def merge(ids, pair, idx):
    newids = []
    i = 0
    while i<len(ids):
        if ids[i] == pair[0] and i < len(ids)-1 and ids[i+1] == pair[1]  :
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BpeTokenizer():
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges_rules = {}  # (int, int) -> int (存储 BPE 合并规则)
        self.vocabulary = {}    #词汇表： {id: bytes}
        self.special_tokens = {}     # 特殊标记: str -> int, e.g. {'<|endoftext|>': 100257}
        #self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""  # 默认分词模式 r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        #   r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        text_utf_sequence = text.encode("utf-8") #text->utf-8
        ids = list(text_utf_sequence) #utf-8->0-255的整数列表

        merges_rules = {} #合并规则(int, int) -> int
        vocabulary = {idx: bytes([idx]) for idx in range(256)} #int->bytes,初始化词汇表

        for i in range (num_merges):
            stats = get_stats(ids)
            pair = max(stats, key = stats.get)
            idx = 256+i

            ids = merge(ids, pair, idx)

            #save the merge
            merges_rules[pair] = idx   #used in encode
            vocabulary[idx] = vocabulary[pair[0]] + vocabulary[pair[1]]  #used in decode


            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocabulary[idx]}) had {stats[pair]} occurrences")
        self.merges_rules = merges_rules # used in encode()
        self.vocabulary = vocabulary   # used in decode()

    def encode(self,text):
        # 1. 预分词
        #words = re.findall(self.pattern, text) if self.pattern else [text] 
    
        text_utf_sequence = text.encode("utf-8")
        ids = list(text_utf_sequence)
        #ids = [list(word.encode("utf-8")) for word in words]  # 每个词转为字节列表

        while len(ids)>=2:
            stats = get_stats(ids)
            pair = min(stats, key = lambda p: self.merges_rules.get(p,float("inf")))#best pair
            if pair not in self.merges_rules:
                break # nothing else can be merged anymore

            idx = self.merges_rules[pair]  # new token 的 id
            ids = merge(ids, pair, idx)  #合并

        return ids

    def decode(self,ids):
        text_byts_sequence = b"".join(self.vocabulary[idx] for idx in ids)
        text = text_byts_sequence.decode("utf-8",errors="replace")
        return text
    
    def save(self, file_prefix):
        """保存tokenizer到文件
        file_prefix: 文件前缀，实际会生成两个文件：
            - file_prefix.model (合并规则和特殊token)
            - file_prefix.vocab (词汇表)
        """
        # 保存合并规则和特殊token
        model_data = {
            'merges_rules': list(self.merges_rules.items()),  # 将元组转换为列表以便JSON序列化
            'special_tokens': self.special_tokens,
            #'pattern': self.pattern
        }
        with open(f"{file_prefix}.model", 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        # 保存词汇表(包含bytes数据，使用pickle)
        with open(f"{file_prefix}.vocab", 'wb') as f:
            pickle.dump(self.vocabulary, f)

    @classmethod
    def load(cls, file_prefix):
        """从文件加载tokenizer"""
        tokenizer = cls()
        
        # 加载合并规则和特殊token
        with open(f"{file_prefix}.model", 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            tokenizer.merges_rules = {tuple(pair): idx for pair, idx in model_data['merges_rules']}  # 转换回元组作为键
            tokenizer.special_tokens = model_data['special_tokens']
            #tokenizer.pattern = model_data['pattern']
        
        # 加载词汇表
        with open(f"{file_prefix}.vocab", 'rb') as f:
            tokenizer.vocabulary = pickle.load(f)
        
        return tokenizer

  



