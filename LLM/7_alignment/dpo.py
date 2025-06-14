from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig

from transformers import TrainingArguments

from trl import DPOTrainer


model_id = "your_model"
csv_dataset = "your_dataset"


prompt_length = 1024
max_seq_length = 1512

# 데이터셋 로드 모델 챗 템플릿 적용 
def create_conversation(sample):
    return {
        "text": tokenizer.apply_chat_template([
            {"role": "system", "content": sample["system"]},  
            {"role": "user", "content": sample["user"]},
            {"role": "assistant", "content": sample["assistant"]}
        ], tokenize=False, add_generation_prompt = False)
        }

    

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data_csv = load_dataset("csv", data_files=csv_dataset)
    data_csv = data_csv['train'] 
    dataset_csv = data_csv.map(csv_create_conversation, remove_columns=data_csv.features, batched=False)
    dataset_csv.shuffle()
    dataset_csv = dataset_csv.train_test_split(test_size=0.1)

    train_dataset = dataset_csv['train']
    eval_dataset = dataset_csv['test']

    # BitsAndBytesConfig int4 설정 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )


    # flash attention 설정 
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16


    # model, tokenizer 로드 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="balanced",
        use_cache=False,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    # pad_token=eos_token으로 해주면 좀 더 좋아진다고 합니다.
    tokenizer.pad_token = tokenizer.eos_token

    # 패딩 사이드를 left로 해야지 성능이 더 좋다 알려져 있습니다. 
    tokenizer.padding_side = 'left'
    # 지난 생성된 답변 자르기에 유용합니다. 
    tokenizer.truncation_side = 'left'


    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        output_dir = "your_dir",
        num_train_epochs=4,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=5e-5,
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=100,
        save_total_limit=20,
        evaluation_strategy="steps",
        eval_steps=30000,
        bf16=True,
        tf32=True,
        push_to_hub=False,
        report_to="mlflow",
    )

    dpo_args = {
        "beta": 0.1,
        "loss_type": "sigmoid"
    }

    trainer = DPOTrainer(
        model,
        ref_model=None,
        peft_config=peft_config, # lora가 있다면
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=prompt_length,
        beta=dpo_args["beta"], # implicit reward에 대한 하이퍼파라미터 : DPO loss의 tempearture
        loss_type=dpo_args["loss_type"] # DPO loss의 종류 선택 (hinge, ipo) 개선된 loss가 계속 나옴
    )

    # 훈련 시작 
    trainer.train()

    # 모델 저장 
    trainer.save_model()

