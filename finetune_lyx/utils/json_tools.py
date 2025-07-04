import json

def load_json(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_txt(data_path):
    with open(data_path, "r") as f:
        data = f.read()
    return data

def write_json(data: dict, file_path: str, file_name: str) -> None:
    if not file_path.endswith('/'):
        file_path += '/'
    with open(file_path + file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
