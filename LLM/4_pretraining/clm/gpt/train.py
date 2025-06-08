import os
import sys
import time
import re
import argparse
import math
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.distributed as dist

import wandb
from dataclasses import asdict

from config import GPTConfig, TrainConfig 
from model import GPT

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
        is_master = True
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

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def set_torch_backends_ampere():
    """
    Ampare architecture : 30xx, a100, h100,..
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def get_run_name(train_cfg):
    batch_size = f"bs{train_cfg.batch_size}"
    learning_rate = f"lr{train_cfg.learning_rate}"
    date = time.strftime("%m%d")
    return f"gpt_{batch_size}_{learning_rate}_{date}"

@torch.no_grad()
def estimate_loss(model, eval_iters, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':
    train_cfg, gpt_cfg = TrainConfig(), GPTConfig() 
    # ============================ Distributed Setting ============================
    init_distributed_mode(train_cfg)


    # =========================== Basic Setting ===============================
    fix_random_seeds(train_cfg.seed+train_cfg.rank)

    train_cfg.tokens_per_iter = train_cfg.gradient_accumulation_steps * train_cfg.world_size * train_cfg.batch_size * train_cfg.block_size

    set_torch_backends_ampere()

    if train_cfg.wandb_log and train_cfg.is_master:
        run_name = get_run_name(train_cfg)
        wandb.login(key=train_cfg.wandb_key, host=train_cfg.wandb_host)
        run = wandb.init(
            project=train_cfg.wandb_project_name,
            entity=train_cfg.wandb_entity,
            config={
                "GPTConfig": asdict(gpt_cfg),
                "TrainConfig": asdict(train_cfg),
            },
            name=run_name
        )

    iter_num = 0
    best_val_loss = 1e9
    # ============================ Dataset =========================================
    if train_cfg.is_master:
        os.makedirs(train_cfg.out_dir, exist_ok=True)

    data_dir = os.path.join('./data', train_cfg.dataset)

    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        """
        np.memmap() 은 헤더만 메모리에 올라가므로 “메모리 소모” 걱정은 적지만,
        OS-레벨 파일 디스크립터 열기 + 내부 캐시 생성 비용이 매호출마다 발생합니다.

        GPU가 학습을 기다리는 시간(I/O stalls)이 크게 증가
        """
        data_array = train_data if split=='train' else val_data

        ix = torch.randint(len(data_array) - train_cfg.block_size, (train_cfg.batch_size,))

        x = torch.stack(
            [torch.from_numpy((data_array[i:i+train_cfg.block_size]).astype(np.int64)) for i in ix]
        )

        y = torch.stack(
            [torch.from_numpy((data_array[i+1:i+1+train_cfg.block_size]).astype(np.int64)) for i in ix]
        )

        if train_cfg.device.type == 'cuda':
            x, y = x.pin_memory().to(train_cfg.device, non_blocking=True), y.pin_memory().to(train_cfg.device, non_blocking=True)
        else:
            x, y = x.to(train_cfg.device), y.to(train_cfg.device)

        return x, y


    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f'found vocab_size = {meta_vocab_size} (inside {meta_path})')

    # ============================ Models =========================================
    if train_cfg.init_from == 'scratch':
        print("Initializing a new model from scratch")

        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        
        gpt_cfg.vocab_size = meta_vocab_size if meta_vocab_size is not None else 50304
        model = GPT(gpt_cfg)

    elif train_cfg.init_from == 'resume':
        print(f"Resuming training from {train_cfg.out_dir}")

        ckpt = os.path.join(train_cfg.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt, map_location=train_cfg.device)
        checkpoint_model_args = checkpoint['model_args']

        for k in ['n_layer', 'n_heads', 'hidden_size', 'block_size', 'bias', 'vocab_size']:
            gpt_cfg[k] = checkpoint_model_args[k]

        model = GPT(gpt_cfg)
        model.load_state_dict(checkpoint['model_state_dict'])

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif train_cfg.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {train_cfg.init_from}")
        override_args = dict(dropout=train_cfg.dropout)
        model = GPT.from_pretrained(train_cfg.init_from, override_args)
        for k in ['n_layer', 'n_heads', 'hidden_size', 'block_size', 'bias', 'vocab_size']:
            gpt_cfg[k] = getattr(model.cfg, k)

    # crop down the model block size if desired, using model surgery
    if gpt_cfg.block_size < model.cfg.block_size:
        model.crop_block_size(gpt_cfg.block_size)

    model.to(train_cfg.device)

    # ============================ Optimizer and Loss ======================================
    optimizer = model.configure_optimizer(
        train_cfg.weight_decay,
        train_cfg.learning_rate,
        (train_cfg.beta1, train_cfg.beta2),
        train_cfg.device
    )

    if train_cfg.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # =========================== Training Setting ===============================
    if train_cfg.dtype == 'float16':
        train_cfg.dtype = torch.float16
    elif train_cfg.dtype == 'bfloat16':
        train_cfg.dtype = torch.bfloat16
    else:
        train_cfg.dtype = torch.float32

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.dtype == 'float16'))

    if train_cfg.world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[train_cfg.local_rank],
            output_device=train_cfg.local_rank
        )

    # =========================== Training loop ====================================
    X, Y = get_batch('train')
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if train_cfg.world_size > 1 else model
    running_mfu = -1.0
    while True:
        if train_cfg.decay_lr:
            lr = get_lr(iter_num, train_cfg.learning_rate, train_cfg.max_iters)
        else:
            lr = train_cfg.learning_rate

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_num % train_cfg.eval_interval == 0 and train_cfg.is_master:
            # print(type(train_cfg.device), train_cfg.device)
            # print(str(train_cfg.device))
            losses = estimate_loss(model, train_cfg.eval_interval, str(train_cfg.device))
            print(f"step {iter_num}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if train_cfg.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu*100
                })
            if losses['val'] < best_val_loss or train_cfg.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': gpt_cfg,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': train_cfg
                    }
                    print(f"saving checkpoint to {train_cfg.out_dir}")
                    torch.save(checkpoint, os.path.join(train_cfg.out_dir, 'ckpt.pt'))

        if iter_num == 0 and train_cfg.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(train_cfg.gradient_accumulation_steps):
            if train_cfg.world_size > 1:
                model.require_backward_grad_sync = (micro_step == train_cfg.gradient_accumulation_steps - 1)

            with torch.amp.autocast(device_type=train_cfg.device.type, dtype=train_cfg.dtype):
                logits, loss = model(X, Y)
                loss = loss / train_cfg.gradient_accumulation_steps

            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        # clip the gradient
        if train_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer) # grad / scale (scale에 곱해진 상태이므로 원래대로 되돌림)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)


        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1


        if iter_num % train_cfg.log_interval == 0 and train_cfg.is_master:
            lossf = loss.item() * train_cfg.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(train_cfg.batch_size * train_cfg.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > train_cfg.max_iters:
            break

    # ============================ Finish ==============================================
    if train_cfg.wandb_log and train_cfg.is_master:
        wandb.finish()
    if train_cfg.is_master:
        dist.destroy_process_group()

