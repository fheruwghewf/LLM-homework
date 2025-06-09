
import json
import argparse
import re
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore

def parse_arguments():
    parser = argparse.ArgumentParser(description="Test a model on a dataset.")
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="response.txt",
        help='Path to the model output.',
        required=True,
    )
    parser.add_argument(
        '--test_set_path',
        type=str,
        default='test.json',
        help='Path to the testing data.',
        required=True
    )
    args = parser.parse_args()
    return args

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

def my_tokenizer(txt: str) -> list[str]:
    temp = txt.split(' ')
    final_list = []
    for word in temp:
        final_list.extend(word.split('-'))
    temp: list[str] = final_list
    final_list = []
    for word in temp:
        final_list.extend(word.split('/'))
    return final_list

def main():
    args = parse_arguments()
    json_data = load_json(args.test_set_path)
    output_data = load_txt(args.model_output_path)

    pattern_user = re.compile(r'USER: (.*?)(?= ASSISTANT: )')
    pattern_assistant = re.compile(r' ASSISTANT: (.*)')

    user = pattern_user.findall(output_data)
    assistant = pattern_assistant.findall(output_data)

    pattern_instruction = re.compile(r'\D+?(?=\d)')
    pattern_input = re.compile(r'\d+')
    instructions = []
    inputs = []
    for i in user:
        instructions.extend(pattern_instruction.findall(i))
        inputs.extend(pattern_input.findall(i))

    correct_counter = 0
    bleu = BLEUScore(n_gram=1)
    bleu_hist = []
    rouge = ROUGEScore(tokenizer=my_tokenizer)
    rouge_hist = []
    rouge_f_hist = []
    rouge_p_hist = []
    rouge_r_hist = []
    for i, alpaca in enumerate(json_data):
        if alpaca['instruction'] == instructions[i] and alpaca['input'] == inputs[i]:
            if alpaca['output'] != assistant[i]:
                score = bleu([assistant[i]], [[alpaca['output']]])
                print('Wrong answer for task ', i, '!!')
                bleu_hist.append(score.item())
                print('BLEU score: ', score.item())
                r_score = rouge(assistant[i], alpaca['output'])
                rouge_hist.append({k:r_score[k].item() for k in r_score.keys()})
                rouge_hist[-1]['target'] = alpaca['output']
                rouge_hist[-1]['output'] = assistant[i]
                rouge_f_hist.append(r_score['rougeL_fmeasure'].item())
                rouge_p_hist.append(r_score['rougeL_precision'].item())
                rouge_r_hist.append(r_score['rougeL_recall'].item())
                # print(r_score)


            else:
                correct_counter += 1
        else:
            print('Data ', i, ' dont match!')

    correct_rate = correct_counter / len(json_data)
    print(f'{correct_counter} of {len(json_data)} correct, which is {correct_rate * 100:.2f}%')
    print('BLEU hist: ', bleu_hist)
    write_json(rouge_hist, 'finetune_lyx/data', 'rouge_hist.json')
    print(rouge_f_hist)
    print(rouge_p_hist)
    print(rouge_r_hist)


if __name__ == "__main__":
    main()