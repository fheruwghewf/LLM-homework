

def text2bin(content: str) -> list[int]:
    return list(content.encode('utf-8'))

def get_frequency(content: list[int]) -> dict:
    freq = {}
    for pair in zip(content, content[1:]):
        freq[pair] = freq.get(pair, 0) + 1
    return freq

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

def find_max_pair(freq: dict) -> tuple[int,int]:
    return max(freq, key=freq.get)

def train(content: str, vocab_size: int) -> tuple[dict, dict]:
    merge_map = {}
    decode_map = {i: i.to_bytes() for i in range(256)}
    
    if vocab_size > 256:
        iter_num = vocab_size - 256
        content_byte = text2bin(content)
        for i in range(iter_num):
            freq = get_frequency(content_byte)
            pair = find_max_pair(freq)
            idx = i + 256
            content_byte = merge(content_byte, pair, idx)
            merge_map[pair] = idx
            decode_map[idx] = decode_map[pair[0]] + decode_map[pair[1]]
    return merge_map, decode_map
