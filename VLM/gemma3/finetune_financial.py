from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig
import torch

# 파이토치 2.7버전 에러 해결
import torch, torch._dynamo
torch._dynamo.config.disable = True        # 전역 OFF

from pathlib import Path
import json

train_prompt_style="""
Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
</think>
{}
"""

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
"""

def formatting_prompts_func(examples):
    inputs = examples["Open-ended Verifiable Question"]
    complex_cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for question, cot, response in zip(inputs, complex_cots, outputs):
        # Append the EOS token to the response if it's not already there
        if not response.endswith(tokenizer.eos_token):
            response += tokenizer.eos_token
        text = train_prompt_style.format(question, cot, response)
        texts.append(text)

    return {"text": texts}

if __name__ == "__main__":

    training_arguments = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        logging_steps=0.2,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        group_by_length=True,
        report_to="none"
    )

    GEMMA_PATH = "/kaggle/input/gemma-3/transformers/gemma-3-4b-it/1"

    # for image-and-text and image-only 
    # 비전-언어 모델
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        # device_map={"": 0}, 
        # device_map="auto",
        attn_implementation="eager" # 어떤 커널로 어텐션 계산할지
    ).to('cuda').eval()
    # eager : 모든 HW/dtype에 대해 100프로 동작. 안정성 최고. gemma2/3 학습에는 eager 권장
    # spda : SPDA 사용, head_dim 제한
    # flash_attention_2 : 가장 빠름. flash-attn 설치 필요. CUDA 11.8+ 필요
    # flex_attention : Pytorch FlexAttention, 다양한 마스크, 편향 조합을 자동으로 최적 커널에 매핑. 아직 버그 있는듯

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

    dataset = load_dataset(
        "TheFinAI/Fino1_Reasoning_Path_FinQA",
        split = "train[0:500]",
        trust_remote_code=True
    )

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
    )

    if training_arguments.process_index == 0:
        question = dataset[0]['Open-ended Verifiable Question']

        inputs = tokenizer(
            [prompt_style.format(question, "") + tokenizer.eos_token],
            return_tensors="pt"
        ).to("cuda")

        print('model device:', model.device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1200,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print('question', question)
        print(response[0].split("### Response:")[1])

    
    model.train()
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



    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=data_collator
    )

    torch.cuda.empty_cache()
    trainer_stats = trainer.train()

    if trainer.is_world_process_zero():

        print("trainer stats:", trainer_stats)


        out_dir = Path(training_arguments.output_dir)
        (out_dir / "train_stats.json").write_text(
            json.dumps(trainer_stats.metrics, indent=2)
        )

        model.eval()
        # Model inference after fine-tuning
        question = dataset[0]['Open-ended Verifiable Question']

        inputs = tokenizer(
            [prompt_style.format(question, "") + tokenizer.eos_token],
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids = inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1200,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print("after fine-tuning question1", question)
        print(response[0].split("### Response:")[1])

        # another question.
        question = dataset[10]['Open-ended Verifiable Question']

        inputs = tokenizer(
            [prompt_style.format(question, "") + tokenizer.eos_token],
            return_tensors="pt"
        ).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1200,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        print("after fine-tuning question2", question)
        print(response[0].split("### Response:")[1])

        # Saving the model and tokenizer
        new_model_local = "Gemma-3-4B-Fin-QA-Reasoning"
        model.save_pretrained(new_model_local)
        tokenizer.save_pretrianed(new_model_local)

        # push to hf hub
        # new_model_online = "ty-kim/Gemma-3-4B-Fin-QA-Reasoning"
        # model.push_to_hub(new_model_online)
        # tokenizer.push_to_hub(new_model_online)




    




