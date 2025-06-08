# config.py --------------------------
from dataclasses import dataclass, asdict
@dataclass
class Config:
    max_len = 128
    batch = 32
    epochs = 3
    lr = 3e-5
    mask_prob = 0.15
    model_name = "klue/bert-base" # "kykim/bert-kor-base" "snunlp/KR-BERT-char16424"
    dataset_name = ("heegyu/kowikitext",)
    gradient_checkpointing = True  # Enable gradient checkpointing for memory efficiency


from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random, math, re
from itertools import chain
from tqdm import tqdm
import os


class MLMCollator:
    def __init__(self, tokenizer, mask_prob=0.15):
        self.vocab_size = tokenizer.vocab_size
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.mask_prob = mask_prob

    def __call__(self, examples):
        batch = torch.tensor([e["input_ids"] for e in examples])
        return self._mask_tokens(batch)
    
    def _mask_tokens(self, inputs):
        labels = inputs.clone()
        # pad 위치 제외
        prob = torch.full_like(inputs, self.mask_prob, dtype=torch.float32)
        prob.masked_fill_(inputs == self.pad_id, 0.0)
        masked = torch.bernoulli(prob).bool()

        # 80% -> [MASK]
        indices_replaced = torch.bernoulli(torch.full_like(inputs, 0.8, dtype=torch.float32)).bool() & masked
        inputs[indices_replaced] = self.mask_id

        # 10% -> 랜덤 토큰
        indices_random = torch.bernoulli(torch.full_like(inputs, 0.5, dtype=torch.float32)).bool() & masked & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long, device=inputs.device)
        inputs[indices_random] = random_words[indices_random]

        labels[~masked] = -100
        return inputs, labels
    
    def _mask_tokens_better(self, inputs):
        labels = inputs.clone()
        prob = torch.rand_like(inputs, dtype=torch.float32) # 0~1 uniform
        masked = prob < self.mask_prob

        # 80% -> [MASK]
        replace_rng = torch.rand_like(prob)
        indices_replaced = (replace_rng < 0.8) & masked
        inputs[indices_replaced] = self.mask_id

        # 10% -> 랜덤 토큰
        random_rng = torch.rand_like(prob)
        indices_random = (random_rng < 0.5) & masked & ~indices_replaced
        random_words = torch.randint(
            self.vocab_size, 
            inputs.shape,
            dtype=torch.long,
            device=inputs.device
        )
        inputs[indices_random] = random_words[indices_random]

        labels[~masked] = -100
        return inputs, labels

    
if __name__ == '__main__':
    cfg = Config()
    print("config:", asdict(cfg))

    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Load dataset
    raw = load_dataset(*cfg.dataset_name, split='train', trust_remote_code=True)
    raw = raw.train_test_split(test_size=0.02, seed=42)
    train_ds, val_ds = raw['train'], raw['test']

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def tokenize(example):
        enc = tokenizer(
            example['text'],
            truncation=True,
            max_length=cfg.max_len,
            padding='max_length'
        )

        return {"input_ids": enc["input_ids"]}


    train_ds = train_ds.map(tokenize, batched=True, num_proc=4, remove_columns=['text'])
    val_ds = val_ds.map(tokenize, batched=True, num_proc=4, remove_columns=['text'])

    collate = MLMCollator(tokenizer, mask_prob=cfg.mask_prob)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False, collate_fn=collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        ).to(device)
    
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency.")

        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_ppl = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total, steps = 0.0, 0
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(input_ids=inputs).logits
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total += loss.item()
            steps += 1
            if steps % 100 == 0:
                pbar.set_postfix(train_loss=total/steps)

        
        model.eval()
        val_loss = 0.0
        vsteps = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(input_ids=inputs).logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
                vsteps += 1

        ppl = math.exp(val_loss / vsteps)
        print(f"Epoch {epoch} - Validation Loss: {val_loss / vsteps:.4f}, PPL: {ppl:.4f}")


        if ppl < best_ppl:
            best_ppl = ppl
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Model saved at {save_dir} with PPL: {best_ppl:.4f}")