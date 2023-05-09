"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import argparse
import os
import math
import sys
import ipdb
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
import datetime
from datasets.distributed import split_dataset_by_node
from torch.utils.data import BatchSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForCausalLM,
    LlamaTokenizer
)
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from datasets import load_dataset,Features,Value
from transformers import DataCollatorForLanguageModeling

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model



def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='6,2,2',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args

def save_hf_format(model, tokenizer, output_dir):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage)
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    set_random_seed(args.seed)
    torch.distributed.barrier()
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path,
                                              fast_tokenizer=True)
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=407,
            padding='max_length',
            return_tensors=None
        )
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    model = create_hf_model(LlamaForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    dataset = load_dataset("json", data_files={"train": "./zh_chs1200-18.json_sc", "test": "./zh_chs1200-18.json_sc"}, streaming = True)
    
    train_dataset, eval_dataset = dataset["train"], dataset["test"]
    train_dataset = split_dataset_by_node(train_dataset, rank = args.global_rank, world_size = torch.distributed.get_world_size())
    eval_dataset = split_dataset_by_node(eval_dataset, rank = args.global_rank, world_size = torch.distributed.get_world_size())
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = 100000
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Micro Batches Per Epochs {num_update_steps_per_epoch}",
            args.global_rank)
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            print(batch)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "epoch", epoch, "step ", step, "rank", args.global_rank,"learning rate", optimizer.get_lr(),  "loss", loss)
            if (step + 1) % 100 == 0:
                print_rank_0(f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****",
                    args.global_rank)
                perplexity = evaluation(model, eval_dataloader)
                print_rank_0(f"ppl: {perplexity}", args.global_rank)
                model.tput_timer.update_epoch_count()
                print_rank_0('saving the final model ... ', args.global_rank)
                pth_save = os.path.abspath(args.output_dir) + "_" + str(epoch) + "_" + str(step) + "_ckpt"
                save_zero_three_model(model, args.global_rank, pth_save, zero_stage=args.zero_stage)
                torch.distributed.barrier()
                model.train()
    print_rank_0('saving the final model ... ', args.global_rank)
    pth_save = os.path.abspath(args.output_dir) + "_" + str(epoch) + "_" + str(step) + "_ckpt"
    save_zero_three_model(model, args.global_rank, pth_save, zero_stage=args.zero_stage)
    torch.distributed.barrier()
    model.train()

if __name__ == "__main__":
    main()