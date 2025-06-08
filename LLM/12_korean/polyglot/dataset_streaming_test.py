from __future__ import annotations
import os, itertools
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer, default_data_collator


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


def build_streaming_dataloader(
        dataset_specs,
        tokenizer,
        seq_len=2048,
        batch_size=1,
        num_workers=2,
        probabilities = None
):
    # 데이터 스트림 열기
    streams = [_open_stream(spec) for spec in dataset_specs]
    
    # interelave : 여러 데이터셋을 섞어서 도메인이 섞이도록 함
    mixed = interleave_datasets(streams, probabilities=probabilities, seed=42)

    # 토크나이즈 (batched=True -> 속도 향상)
    def tok_fn(batch):
        texts = batch["text"] if "text" in batch else list(batch.values())[0]
        # 아직 pytorch tensor로 받지 않는다
        # 순수 파이썬 리스트로 받도록 한다.
        ids = tokenizer(
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

    # seq len 단위로 잘라 패킹
    def group_fn(batch):
        # ① 배치 안 모든 토큰 연결
        concat = list(itertools.chain.from_iterable(batch["ids"]))
        # ② seq_len 배수 길이만 남기기
        total = len(concat) - len(concat) % seq_len
        # ③ 잘라서 새 예제 리스트 생성
        return {"input_ids": [concat[i : i + seq_len]
                            for i in range(0, total, seq_len)]}

        # if cut_len == 0:
        #     return {"input_ids": []}
        # blocks = [flat[i: i + seq_len] for i in range(0, cut_len, seq_len)]
        # return {"input_ids": blocks}
    

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
        batch_size=batch_size,
        collate_fn=default_data_collator,
        num_workers=num_workers,
        pin_memory=True,
    )


import json, itertools, time
def count_tokens(tokenizer, ds_name, subset=None, split="train", ):
    ds = load_dataset(ds_name, subset, split=split, streaming=True)
    total = 0
    t0 = time.time()

    for batch in tqdm(ds.iter(batch_size=1000), desc=f"{ds_name}:{subset or split}"):
        texts = batch["text"] if "text" in batch else list(batch.values())[0]
        token_lists = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        total += sum(len(t) for t in token_lists)

    print(f"{ds_name}:{subset or split} → {total:,} tokens "
          f"({(time.time()-t0)/60:.1f} min)")
    return total

if __name__ == "__main__":
    specs = [
        "allenai/c4:ko",
        "heegyu/kowikitext",
        "heegyu/namuwiki-extracted"
    ]
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b", use_fast=True)
    # GPT 계열은 pad_token이 없어서 None으로 설정되어 있음
    # 추후 필요할 수 있으므로 pad_token을 설정해줌 : 하위호환성
    tokenizer.pad_token = tokenizer.eos_token

    # # ds = load_dataset("allenai/c4", "ko", trust_remote_code=True)
    # dl = build_streaming_dataloader(
    #     specs,
    #     tokenizer,
    #     seq_len=2048,
    #     batch_size=2,
    #     probabilities=[1.0]
    # )

    # # 한 배치 확인
    # batch = next(iter(dl))
    # print("input_ids shape:", batch["input_ids"].shape)  # (B, seq_len)
    # print(batch["input_ids"][0][:20])  # 첫 번째 배치의 첫 20개 토큰 ID 출력
    # print("input_ids dtype:", batch["input_ids"].dtype)  # 데이터 타입 확인


    # batch = next(iter(dl))
    # print("input_ids shape:", batch["input_ids"].shape)  # (B, seq_len)
    # print(batch["input_ids"][0][:20])  # 첫 번째 배치의 첫 20개 토큰 ID 출력
    # print("input_ids dtype:", batch["input_ids"].dtype)  # 데이터 타입 확인

    token_counts = {}
    for spec in specs:
        if ":" in spec:
            name, subset = spec.split(":", 1)
        else:
            name = spec
            subset = None

        tok_num = count_tokens(tokenizer, name, subset)
        token_counts[f"{name}:{subset or 'train'}"] = tok_num

    print(token_counts)
