from huggingface_hub import login
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from transformers import DataCollatorForLanguageModeling
import torch

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments

from dataclasses import dataclass, field
from typing import Optional
import argparse

import time
import wandb

@dataclass
class CustomArguments:
    hf_token: str = field(metadata={"help": "Hugging Face access token."})
    wandb_key: str = field(metadata={"help": "WandB API key."}) 
    model_id: str = field(
        default="Qwen/Qwen3-32B",
        metadata={"help": "Model ID for Hugging Face."}
    )
    wandb_host: str = field(
        default="http://wandb.artfacestudio.com",
        metadata={"help": "WandB host URL."}
    )
    project_name: str = field(
        default="qwen-medical-cot",
        metadata={"help": "WandB project name."}
    )

@dataclass
class MySFTConfig(TrainingArguments):
    output_dir="output"         # directory to save and repository id
    max_seq_length=512                     # max sequence length for model and packing of the dataset
    packing=True                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=1                     # number of training epochs
    per_device_train_batch_size=8          # batch size per device during training
    per_device_eval_batch_size=1
    gradient_accumulation_steps=4          # number of steps before performing a backward/update pass
    gradient_checkpointing=True            # use gradient checkpointing to save memory
    optim="paged_adamw_32bit"              # use fused adamw optimizer
    logging_steps=0.2                       # log every 10 steps
    save_strategy="epoch"                  # save checkpoint every epoch
    learning_rate=2e-4                     # learning rate, based on QLoRA paper
    fp16=False   # use float16 precision
    bf16=True   # use bfloat16 precision
    max_grad_norm=0.3                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant"           # use constant learning rate scheduler
    push_to_hub=False                      # push model to hub
    report_to="wandb"                # report metrics to tensorboard
    logging_strategy="steps"
    group_by_length=True

if __name__ == "__main__":
    parser = HfArgumentParser((CustomArguments, MySFTConfig))
    custom_args, training_args = parser.parse_args_into_dataclasses()

    wandb.login(key=custom_args.wandb_key, host=custom_args.wandb_host)
    wandb.init(
        project=custom_args.project_name,
        name=f"qwen_{time.strftime('%m%d')}",
        config={**vars(custom_args), **training_args.to_dict()},
    )

    hf_token = os.environ.get("HF_TOKEN")
    login(hf_token)

    # nn.Linear를 4비트로 가져오기. 
    # nn.Embedding과 lm_head는 bf16으로 유지
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_dir = "Qwen/Qwen3-32B"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
        )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training( # 4bit, 8bit로 peft할 때는 꼭 하자
        model, 
        use_gradient_checkpointing=False # 여길 True 해놓고 Argument에서 또 True하면 충돌 생김
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    model = get_peft_model(model, peft_config)

    dataset = load_dataset(
        "ty-kim/medical_cot",
        trust_remote_code=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = SFTTrainer(
        model=model, 
        args=training_args,
        train_dataset=dataset['train'], # train만 제대로 넘어가야함
        peft_config=peft_config,
        data_collator=data_collator,
    )

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    trainer.train()
    wandb.finish()

    # Save the final model again to the Hugging Face Hub
    # 그냥 save 하면 adapter만 저장됨
    # trainer_output 폴더에는 adapter만 저장
    trainer.save_model()

    # 만약에 full model과 같이 저장하고 싶다면
    # 다음처럼 merge_and_unload()를 사용
    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    # Load Model base model
    model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(custom_args.model_id, low_cpu_mem_usage=True)

    # Merge LoRA and base model and save
    # merged_model에 adapter가 merge된 모델이 저장됨
    peft_model = PeftModel.from_pretrained(model, training_args.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

    processor = AutoTokenizer.from_pretrained(training_args.output_dir)
    processor.save_pretrained("merged_model")


    # new_model_name = "Qwen-3-32B-Medical-Reasoning"
    # model.push_to_hub(new_model_name)
    # tokenizer.push_to_hub(new_model_name)