import argparse
import os
import sys
import inspect
import wandb
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
import itertools

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger

from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    default_data_collator,
    get_linear_schedule_with_warmup,
)


from dataclasses import dataclass, field, asdict
@dataclass
class TrainConfig:
    # wandb
    wandb_key = None
    wandb_host = ""
    wandb_project = ""
    wandb_run_name = ""

    # 모델
    model_name: str = "EleutherAI/polyglot-ko-1.3b"

    # 데이터
    dataset_specs: List[str] = field(default_factory=lambda: ["allenai/c4:ko", "heegyu/kowikitext", "heegyu/namuwiki-extracted"])
    val_specs: Tuple[str, ...] = ("teknium/ko-mmlu:validation", "beomi/kobench:test", "squad_kr:validation") # mmlu, kobench, korquad
    sequence_length: int = 2048
    
    # 학습
    batch_size: int = 16
    grad_accum: int = 1 # grad accum 마이크로 스텝 단위 (1이면 사용 x)
    num_workers: int = 8
    num_epochs: int = 10
    warmup_steps: int = 1000
    warmup_ratio: float = 0.01
    learning_rate: float = 2e-5
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42

    # Fabric
    pl_strategy = 'deepspeed_stage_2'
    pl_precision: str = "bf16-mixed"
    
    # 기타
    save_interval: int = 1000
    output_dir: str = "./output"
    logging_steps: int = 10
    save_steps: int = 5000

def _open_stream(spec, streaming=True):
    if os.path.isdir(spec):
        return load_dataset(
            "text",
            data_files={"train": str(Path(spec) / "*.txt")},
            split="train",
            streaming=streaming
        )
    
    # "name:split" 혹은 "name:subset" 패턴 처리
    if ":" in spec:
        name, suffix = spec.split(":", 1)
        # split 지정만 있는 경우
        if suffix in {"train", "validation", "test"}:
            return load_dataset(name, split=suffix, streaming=streaming)
        # subset 지정 split은 기본 train
        return load_dataset(name, name=suffix, split="train", streaming=streaming)
    
    return load_dataset(spec, split="train", streaming=streaming)

from torch.optim.lr_scheduler import LambdaLR
import math

def build_warm_cos_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # cosine
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

class PolyglotTrainer:
    def __init__(self, fabric, cfg):
        self.fabric = fabric
        self.cfg = cfg

        try:
            self.logger = self.fabric.logger
        except Exception as e:
            print(e)

        self._init_seed()
        self._init_weight_dtype()
        self._build_tokenizer()
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self.setup_fabric()


    def _init_seed(self):
        if self.cfg.seed:
            torch.manual_seed(self.cfg.seed)
            self.fabric.seed_everything(self.cfg.seed)

    def _init_weight_dtype(self):
        precision_str = self.cfg.pl_precision
        if "16" in precision_str or "transformer-engine" in precision_str:
            if "bf" in precision_str:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def _build_tokenizer(self):
        tok = AutoTokenizer.from_pretrained(self.cfg.model_name, use_fast=True)
        tok.pad_token = tok.eos_token
        tok.model_max_length = self.cfg.sequence_length
        self.tokenizer = tok

    def _stream_to_loader(self, specs: List[str], batch_size: int):
        specs = self.cfg.dataset_specs

        streams = [_open_stream(spec) for spec in specs]

        mixed = interleave_datasets(
            streams,
            stopping_strategy="all_exhausted",
            seed=cfg.seed
        )

        def tok_fn(batch):
            texts = batch["text"] if "text" in batch else list(batch.values())[0]
            ids = self.tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False
            )["input_ids"]

            return {"ids": ids}

        tokenized = mixed.map(
            tok_fn,
            batched=True,
            remove_columns=[c for c in mixed.column_names if c != "text"]  # 텍스트 열만 제거
        )

        def group_fn(batch):
            concat = list(itertools.chain.from_iterable(batch["ids"]))
            total = len(concat) - len(concat) % self.cfg.sequence_length

            return {"input_ids": [concat[i: i + self.cfg.sequence_length]
                    for i in range(0, total, self.cfg.sequence_length)]}

        grouped = tokenized.map(
            group_fn, 
            batched=True, 
            batch_size=1000, 
            remove_columns=["ids"]          # ids 열만 삭제
        ).filter(lambda x: x["input_ids"] is not None)  # 빈 예제 제거

        # 텐서 자동 변환
        grouped = grouped.with_format("torch")

        # 최종 DataLoader
        return DataLoader(
            grouped,
            batch_size=self.cfg.batch_size,
            collate_fn=default_data_collator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=False
        )

    def _build_dataloader(self):
        self.train_loader = self._stream_to_loader(
            specs = self.cfg.dataset_specs,
            batch_size = self.cfg.batch_size
        )

        self.val_loader = self._stream_to_loader(
            specs = self.cfg.val_specs,
            batch_size = self.cfg.batch_size
        )

    def _build_model(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name, 
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
        )

    def _build_optimizer(self):

        params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [
            {'params': params, 'lr': self.cfg.learning_rate}
        ]

        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        extra_args = dict(fused=True) if fused_available and torch.cuda.is_available() else dict()
        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(self.cfg.beta1, self.cfg.beta2),
            weight_decay = self.cfg.weight_decay,
            **extra_args
        )

        total_updates = (len(self.train_loader) * self.cfg.num_epochs) // self.cfg.grad_accum
        warm_steps = int(total_updates * self.cfg.warmup_ratio)

        self.scheduler = build_warm_cos_scheduler(self.optimizer, total_updates, warm_steps)
    def setup_fabric(self):

        for name in ("train_loader", "val_loader", "test_loader"):
            loader = getattr(self, name, None)
            if loader is not None:
                setattr(self, name, self.fabric.setup_dataloaders(loader))

        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.scheduler = self.fabric.setup_lr_scheduler(self.scheduler)
        self.use_accum = self.cfg.grad_accum > 1


    def train(self):
        self.model.train()
        self.num_train_step = 0

        for epoch in range(self.cfg.num_epochs):
            self.train_loader = tqdm(self.train_loader, desc="Training") if self.fabric.is_global_zero else self.train_loader
            for idx, batch in enumerate(self.train_loader):
                sync_ctx = (self.fabric.no_backward_sync(self.model) if self.use_accum and (idx % self.cfg.grad_accum != 0) else self.fabric.enable_grad_sync())

                with sync_ctx:
                    out = self.model(**batch, labels=batch["input_ids"])
                    if self.use_accum:
                        loss = out.loss / self.cfg.grad_accum
                    else:
                        loss = out.loss

                    self.fabric.backward(loss, model=self.model)

                if (not self.use_accum) or (idx % self.cfg.grad_accum == 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if self.fabric.is_global_zero and self.num_train_step % 20 == 0:
                        self.fabric.print(f'step {self.num_train_step:>6} loss {loss.item():.4f}')
                        self.fabric.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=self.num_train_step)

                    
                    if self.fabric.is_global_zero and self.num_train_step % self.cfg.save_interval == 0 and self.num_train_step != 0:
                        ckpt = Path(self.cfg.output_dir) / f"step_{self.num_train_step}"
                        ckpt.mkdir(parents=True, exist_ok=True)
                        self.fabric.save(os.fspath(ckpt / "model.bin"), self.model)

                
                self.num_train_step += 1

            self.fabric.print(f"epoch {epoch+1} finished")
            self.validate()

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        losses = []
        for batch in self.val_loader:
            out = self.model(**batch, labels=batch["input_ids"])
            losses.append(out.loss.float())

        ppl = torch.exp(torch.stack(losses).mean())
        self.fabric.print(f"val ppl {ppl:.2f}")
        self.fabric.log({"val_ppl": ppl.item()}, step=self.num_train_step)
        self.model.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with Polyglot")
    parser.add_argument('--nnodes', type=int, default=1)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args() 

    cfg = TrainConfig()


    wandb_logger = None
    if cfg.wandb_key:
        wandb.login(key=cfg.wandb_key, host=cfg.wandb_host)
        wandb_logger = WandbLogger(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg)
        )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    fabric = Fabric(
        accelerator='cuda',
        num_nodes=args.nnodes,
        devices=args.ngpus,
        strategy=cfg.pl_strategy,
        precision=cfg.pl_precision,
        loggers=wandb_logger
    )

    fabric.launch()
    trainer = PolyglotTrainer(fabric, cfg)
    trainer.train()