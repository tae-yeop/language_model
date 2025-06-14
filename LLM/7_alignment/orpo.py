import gc 
import os 
import pandas as pd 
import random 


import torch
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk, concatenate_datasets 
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from utils import find_all_linear_names, print_trainable_parameters
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

import mlflow


mlflow.enable_system_metrics_logging()
mlflow.autolog()

# dataset load and transform csv -> hugging face style
def return_prompt_and_responses(samples):
    return {
        "prompt": [f"### Input:{system} {question}\n ### Output:"
                  for system, question in zip(samples['system'], samples['question'])],
        "chosen": samples['chosen'],
        "rejected": samples['rejected'],
    }

if __name__ == '__main__':
    ## 데이터셋을 이용한 모델은 다 상업적 이용 가능
    df = pd.read_csv(dataset_name, encoding='utf-8-sig')
    df = df.fillna('')
    data = Dataset.from_pandas(df)
    column_names = data.column_names

    datasets = data.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=column_names,
    )

    # 테스트 사이즈 10%
    datasets = datasets.train_test_split(test_size=0.01)

    # flash attention 
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16

    # QLoRA config 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    max_memory = {i: '45800MB' for i in range(torch.cuda.device_count())}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_mobel)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_mobel,
        quantization_config=bnb_config,
        attn_implementation=attn_implementation,
        device_map = "sequential",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model, tokenizer = setup_chat_format(model, tokenizer)
    model = prepare_model_for_kbit_training(model)

        
    # Lora config
    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM" ,
        target_modules=find_all_linear_names(model),
    )

    #  orp training 
    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        lr_scheduler_type="cosine",
        max_length=2048,
        max_prompt_length=1024,
        beta=0.1,
        bf16=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        num_train_epochs=5,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=10,
        warmup_ratio=0.05,
        output_dir=output_dir,
        max_steps=10000,
    )


    orpo_trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )


    orpo_trainer.train()
    orpo_trainer.save_model(output_dir)


    checkpoint_name = "your_check_point"
    output_dir = os.path.join(output_dir, checkpoint_name)
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)