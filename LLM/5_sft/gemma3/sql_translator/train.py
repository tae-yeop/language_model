import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig, HfArgumentParser
from peft import LoraConfig
from trl import SFTConfig
from trl import SFTTrainer
from peft import PeftModel
from dataclasses import dataclass, field
from typing import Optional
import argparse
import time
import wandb


@dataclass
class CustomArguments:
    # 기본값 없는걸 앞에 몰아서 써야함. 안그럼 에러
    # 이게 싫으면 모든 항목에 기본값을 deafualt=None을 주도록 함
    hf_token: str = field(metadata={"help": "Hugging Face access token."})
    wandb_key: str = field(metadata={"help": "WandB API key."}) 
    model_id: str = field(
        default="google/gemma-3-1b-pt",
        metadata={"help": "Model ID for Hugging Face."}
    )
    wandb_host: str = field(
        default="http://wandb.artfacestudio.com",
        metadata={"help": "WandB host URL."}
    )
    project_name: str = field(
        default="gemma-text-to-sql",
        metadata={"help": "WandB project name."}
    )

@dataclass
class MySFTConfig(SFTConfig):
    output_dir="gemma-text-to-sql"         # directory to save and repository id
    max_seq_length=512                     # max sequence length for model and packing of the dataset
    packing=True                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=6                     # number of training epochs
    per_device_train_batch_size=12          # batch size per device during training
    gradient_accumulation_steps=4          # number of steps before performing a backward/update pass
    gradient_checkpointing=True            # use gradient checkpointing to save memory
    optim="adamw_torch_fused"              # use fused adamw optimizer
    logging_steps=10                       # log every 10 steps
    save_strategy="epoch"                  # save checkpoint every epoch
    learning_rate=2e-4                     # learning rate, based on QLoRA paper
    fp16=False   # use float16 precision
    bf16=True   # use bfloat16 precision
    max_grad_norm=0.3                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant"           # use constant learning rate scheduler
    push_to_hub=False                      # push model to hub
    report_to="wandb"                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }


if __name__ == "__main__":
    parser = HfArgumentParser((CustomArguments, MySFTConfig))
    custom_args, training_args = parser.parse_args_into_dataclasses()

    wandb.login(key=custom_args.wandb_key, host=custom_args.wandb_host)

    wandb.init(
        project=custom_args.project_name,
        name=f"gemma3_{time.strftime('%m%d')}",
        config={**vars(custom_args), **training_args.to_dict()},
    )

    dataset = load_dataset("ty-kim/sql_translator", token=custom_args.hf_token)
    train_dataset = dataset["train"].shuffle().select(range(len(dataset["train"])))


    if custom_args.model_id == "google/gemma-3-1b-pt":
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForImageTextToText


    # Check if GPU benefits from bfloat16
    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
    

    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
        device_map=None # Trainer+DDP 조합이면 일단 걍 CPU에 올리기
    )

    # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )

    # Load model and tokenizer
    model = model_class.from_pretrained(custom_args.model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
    )

    # Create Trainer object
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer
    )

    # Start training, the model will be automatically saved to the Hub and the output directory
    trainer.train()

    # Save the final model again to the Hugging Face Hub
    # 그냥 save 하면 adapter만 저장됨
    # trainer_output 폴더에는 adapter만 저장
    trainer.save_model()

    wandb.finish()

    # 만약에 full model과 같이 저장하고 싶다면
    # 다음처럼 merge_and_unload()를 사용
    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    # Load Model base model
    model = model_class.from_pretrained(custom_args.model_id, low_cpu_mem_usage=True)

    # Merge LoRA and base model and save
    # merged_model에 adapter가 merge된 모델이 저장됨
    peft_model = PeftModel.from_pretrained(model, training_args.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

    processor = AutoTokenizer.from_pretrained(training_args.output_dir)
    processor.save_pretrained("merged_model")