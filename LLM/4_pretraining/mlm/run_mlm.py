import sys
import os
import time
from itertools import chain
from pathlib import Path
import re
import math
import random
import argparse
import wandb
import numpy as np

import datasets
from datasets import load_dataset

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_torch_backends_ampere():
    """
    Ampare architecture : 30xx, a100, h100,..
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def get_run_name(args):
    model_name = f"{args.model_name}"
    batch_size = f"bs{args.batch_size}"
    learning_rate = f"lr{args.learning_rate}"
    date = time.strftime("%m%d")
    return f"mlm_{model_name}_{batch_size}_{learning_rate}_{date}"


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def resolve_root_node_address(nodes: str) -> str:
    """The node selection format in SLURM supports several formats.

    This function selects the first host name from

    - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
    - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
    - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root

    """
    nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  # Take the first node of every node range
    nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)  # handle special case where node range is single number
    return nodes.split(" ")[0].split(",")[0]

def get_main_address():
    root_node = os.environ.get("MASTER_ADDR")
    if root_node is None:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = resolve_root_node_address(nodelist)
        os.environ["MASTER_ADDR"] = root_node

    return root_node

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 실행시 torchrun or torch.distributed.launch --
        args.rank = int(os.environ["RANK"]) # dist.get_rank()
        args.world_size = int(os.environ['WORLD_SIZE']) # dist.get_world_size()
        args.local_rank = int(os.environ['LOCAL_RANK']) # args.rank % torch.cuda.device_count()
    elif 'SLURM_PROCID' in os.environ:
        # 스크립트 실행시
        # #SBATCH --gres=gpu:x
        # #SBATCH --ntasks-per-node=x
        # python train.py
        # python train.py를 x번 돌리는 경우
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.node_rank = int(os.environ['SLURM_NODEID'])

        os.environ['WORLD_SIZE'] = str(args.world_size)
        os.environ['RANK'] = str(args.rank)

    elif torch.cuda.is_available():
        # 스크립트 실행시 
        # 슬럼 옵션 사용하지 않을시
        # 이때는 torchrun or torch.distributed.launch도 안쓴다고 가정
        print('Will run the code on one GPU.')
        args.rank, args.local_rank, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        args.is_master = True
    else:
        # GPU 마저도 안쓰는 경우 그냥 종료
        print('Not using distributed mode')
        sys.exit(1)

    os.environ['MASTER_ADDR'] = get_main_address()
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    args.is_master = args.rank == 0
    dist.barrier()

    setup_for_distributed(args.rank == 0)
    print('| distributed init (rank {}): {}'.format(args.rank, os.environ['MASTER_ADDR']), flush=True)


# from dataclasses import dataclass
# @dataclass
# class Config():
#     wandb_key

def get_dataset(args):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        args.max_seq_length = tokenizer.model_max_length
        if args.max_seq_length > 1024:
            args.max_seq_length = 1024
    else:
        args.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    overwrite_cache = args.overwrite_cache and args.rank == 0
    load_from_cache = not overwrite_cache


    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )
        
       
        if args.rank > 0:
            print("Waiting for main process to perform the mapping")
            torch.distributed.barrier()
        
        # 토크나이즈 결과를 디스크 캐시(arrow 파일)에 write
        # 같은 파라미터로 실행하면 캐시만 읽어서 빠르게 넘어감
        tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=overwrite_cache if args.is_master else load_from_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    
        if args.rank == 0:
            print("Loading results from main process")
            torch.distributed.barrier()

    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        
        if args.rank > 0:
            print("Waiting for main process to perform the mapping")
            torch.distributed.barrier()

        tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=overwrite_cache if args.is_master else load_from_cache,
                desc="Running tokenizer on every text in dataset",
            )
        
        if args.rank == 0:
            print("Loading results from main process")
            torch.distributed.barrier()

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

            first_key = list(examples.keys())[0]
            total_length = len(concatenated_examples[first_key])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result
        

        if args.rank > 0:
            print("Waiting for main process to perform the mapping")
            torch.distributed.barrier()

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=overwrite_cache if args.is_master else load_from_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )

        if args.rank == 0:
            print("Loading results from main process")
            torch.distributed.barrier()


    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    return train_dataset, eval_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--validation_split_percentage", default=5)
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    parser.add_argument("--model_name", type=str, default=None,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # ============================ Distributed Setting ============================
    init_distributed_mode(args)


    # ============================ Basic Setting ==================================
    fix_random_seed(args.seed+args.rank)

    set_torch_backends_ampere()


    if args.wandb and args.is_master:
        run_name = get_run_name(args)
        wandb.login(key=args.wandb_key, host=args.wandb_host, force=True)
        run = wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity,
            config=
            {"Config" : args},
            name=run_name)


    # ============================ Dataset =========================================
    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=True
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=True
            )

            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=True
            )

    # ============================ Tokenizer =========================================
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True
    )


    train_dataset, eval_dataset = get_dataset(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    g = torch.Generator()
    g.manual_seed(args.seed + args.rank)

    train_sampler = DistributedSampler(
        train_dataset,
        rank=args.rank,
        num_replicas=args.world_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )

    
    eval_sampler = DistributedSampler(
        eval_dataset,
        rank=args.rank,
        num_replicas=args.world_size,
        shuffle=False
    )

    eval_loader = DataLoader(
        eval_sampler,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        generator=g,
    )

    # ============================ Models =========================================
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,

    )

    model.to(args.device)
    if args.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    if args.compile:
        model = torch.compile(
            model,
            mode="default", # 또는 "reduce-overhead", "max-autotune"
            backend="inductor"
            )
    # ============================ Training =========================================
    scaler = torch.cuda.amp.GradScaler(enabled=)
    starting_epoch = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        if args.with_tracking:
            total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**batch)
                loss = outputs.loss
            if args.with_tracking:
                total_loss += loss.detach().float()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)



        model.eval()
        losses = []
        for step, batch in enumerate(eval_loader):
            with torch.no_grad():
                outputs = model(**batch)
            
            loss = outputs.loss
            losses.append(loss)

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except:
            perplexity = float("inf")

