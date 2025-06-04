import json
import argparse
import torch
import transformers

from utils import str2bool
from model import get_model_and_tokenizer
from dataset import get_test_dataloader
from generate import generate_response


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to the training data.',
        required=True
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=1024,
        help='The maximum sequence length of the model.',
    )
    parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )
    parser.add_argument(
        "--use_lora", 
        type=str2bool, 
        default=False, 
        help="Whether to use LoRA.")
    parser.add_argument(
        '--lora_dim',
        type=int,
        default=8,
        help='LoRA dimension.',
    )
    parser.add_argument(
        '--lora_scaling',
        type=int,
        default=32,
        help='LoRA scaling.',
    )
    parser.add_argument(
        "--lora_module_name", 
        type=str, 
        default="h.", 
        help="The scope of LoRA.")
    parser.add_argument(
        '--lora_load_path',
        type=str,
        default=None,
        help='The path to saved LoRA checkpoint.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    parser.add_argument(
        '--use_cuda',
        type=str2bool,
        default=True,
        help='Whether to use CUDA.',
    )
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='test',
        help='Where to store output files.',
    )
    args = parser.parse_args()
    return args

def test(model: transformers.GPT2LMHeadModel, test_dataloader: torch.utils.data.DataLoader, device: str):
    model.eval()
    test_loss = []
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        test_loss.append(outputs.loss.item())
    model.train()
    return test_loss

def get_instruction(data_path: str) -> list[str]:
    with open(data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    instruction = []
    alpaca: dict
    for alpaca in data:
        instruction.append(alpaca['instruction'] + alpaca['input'])
    
    return instruction

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
    else:
        device = "cpu"
    lora_args = None
    if args.use_lora:
        lora_args = {"part_module_name": args.lora_module_name, "lora_dim": args.lora_dim, "lora_scaling": args.lora_scaling, "lora_load_path": args.lora_load_path}
    model, tokenizer = get_model_and_tokenizer(args.model_name_or_path, args.trust_remote_code, args.max_length, args.use_lora, lora_args)
    model: transformers.GPT2LMHeadModel
    model.to(device)

    test_loader = get_test_dataloader(tokenizer, args.data_path)
    loss = test(model, test_loader, device)
    print(loss)

    instructions = get_instruction(args.data_path)
    generate_response(model, tokenizer, instructions, device, args.output_dir_name)
    
if __name__ == "__main__":
    main()
