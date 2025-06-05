
import json
import argparse
import re

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
    for i, alpaca in enumerate(json_data):
        if alpaca['instruction'] == instructions[i] and alpaca['input'] == inputs[i]:
            if alpaca['output'] != assistant[i]:
                print('Wrong answer for task ', i, '!!')
            else:
                correct_counter += 1
        else:
            print('Data ', i, ' dont match!')

    correct_rate = correct_counter / len(json_data)
    print(f'{correct_counter} of {len(json_data)} correct, which is {correct_rate * 100:.2f}%')

if __name__ == "__main__":
    main()