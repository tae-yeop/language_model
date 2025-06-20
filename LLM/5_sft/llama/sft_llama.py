if __name__ == "__main__":

    # QLoRA를 위한 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, # 선형 레이어의 nf4또는 FP4 로 대체하여 4비트 양자화 설정
        bnb_4bit_quant_type="nf4", # nf4 양자화
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, # 모델의 경로, 이름
        torch_dtype=torch.bfloat16, 
        quantization_config=bnb_config, 
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)  

    peft_config = LoraConfig(
        r=128, # LoRA Rank, 데이터셋이 복잡할수록 높은 값이 필요
        lora_alpha=16, # scailing factor, 낮으면 기존 데이터를 위주로, 높으면 새로운 데이터 위주로 받아들이게 하는 요소입니다. 8, 16정도로 사용
        target_modules=find_all_linear_names(model), # 훈련할 특성 가중치와 행렬을 결정
        lora_dropout=0.05, # LoRA의 dropout 확률
        bias="none", # LoRA의 bias로 'none', 'all', 'lora_only'
        task_type="CAUSAL_LM" # 'CAUSAL_LM', 'FEATURE_EXTRACTION', 'QUESTION_ANS', 'SEQ_2_SEQ_LM' , 'SEQ_CLS', 'TOKEN_CLS'등
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)


    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=15,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3, # 오래된 체크포인트 순으로 삭제,3 => 가장 최신화된 3개
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_steps=8000,
    )

    trainer = SFTTrainer(
        model,
        train_dataset=datasets,
        dataset_text_field="text", # 데이터셋에서 트레이닝 대상이 되는 컬럼
        tokenizer=tokenizer,
        max_seq_length=4096, # 한번에 입력 받는 최대한의 길이, 커질수록 gpu 메모리 사용량 증가
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "llama2_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

