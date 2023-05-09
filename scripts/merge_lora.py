from dataclasses import dataclass, field
from typing import Optional
import argparse
import peft
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

parser = argparse.ArgumentParser(description='Merge base model and LoRA using Peft')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to the base model')
parser.add_argument('--lora', type=str, required=True,
                    help='Path to the LoRA adapter')
parser.add_argument('--output', type=str, required=True,
                    help='Path to save the merged model')
args = parser.parse_args()

peft_config = PeftConfig.from_pretrained(args.lora)
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the LoRA model
model = PeftModel.from_pretrained(model, args.lora)
model.eval()

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

model.save_pretrained(args.output)