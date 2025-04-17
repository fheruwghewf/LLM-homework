
class Tokenizer:
    def __init__(self):
        self.merge_map: dict = {}
        self.decode_map: dict = {}

    def train(self, text: str, vocab_size: int) -> None:
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        merge_map = {}
        decode_map = {i: i.to_bytes() for i in range(256)}
        
        if vocab_size > 256:
            iter_num = vocab_size - 256
            text_byte = self.text2bin(text)
            for i in range(iter_num):
                freq = self.get_frequency(text_byte)
                pair = self.find_max_pair(freq)
                idx = i + 256
                text_byte = self.merge(text_byte, pair, idx)
                merge_map[pair] = idx
                decode_map[idx] = decode_map[pair[0]] + decode_map[pair[1]]

        # return merge_map, decode_map
        self.merge_map = merge_map
        self.decode_map = decode_map

    def encode(self, text: str) -> list[int]:
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        ids = self.text2bin(text)
        while len(ids) > 1:
            pair = min(ids, key=lambda p:self.merge_map.get(p, len(self.merge_map)))
            if pair not in self.merge_map:
                # nothing to merge
                break
            else:
                ids = self.merge(ids, pair, self.merge_map[pair])

        return ids


    def decode(self, ids: list[int]) -> str:
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        return self.bin2text(b''.join([self.decode_map[i] for i in ids]))

    @staticmethod
    def text2bin(content: str) -> list[int]:
        return list(content.encode('utf-8'))
    
    @staticmethod
    def bin2text(content: bytes) -> str:
        return content.decode('utf-8', errors='replace')

    @staticmethod
    def get_frequency(content: list[int]) -> dict:
        freq = {}
        for pair in zip(content, content[1:]):
            freq[pair] = freq.get(pair, 0) + 1
        return freq

    @staticmethod
    def merge(content: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        i = 0
        new_content = []
        while i < len(content):
            if i < len(content) - 1:
                if (content[i], content[i+1]) == pair:
                    new_content.append(idx)
                    i += 2
                    continue
            new_content.append(content[i])
            i += 1

        return new_content

    @staticmethod
    def find_max_pair(freq: dict) -> tuple[int,int]:
        return max(freq, key=freq.get)