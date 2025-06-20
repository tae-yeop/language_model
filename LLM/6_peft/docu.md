모델 전체 weight를 학습하는게 아니라 적은 파라미터만 학습




QLoRA

파인튜닝할 때 두루두루 적용할 수 있다


BitsAndBytesConfig를 써서 4비트로 로드하고 LoRA 어댑터만 학습하기
이러면 65B 모델도 48GB GPU에서 학습가능.

QLora 기본
```
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,          # 8-bit 경로는 자동 무시
)
```

속도 최적화 실험

llm_int8_threshold 값을 6.0 → 4.8 → 3.0 … 처럼 낮추며 지표 모니터링.

4-bit라면 compute_dtype를 fp16 대신 bf16 으로 두면 Ampere/Hopper GPU에서 속도 손해 없이 안정.

### QLoRA

- 효율적인 파인튜닝 테크닉
- 사전에 학습한 LLM을 4bit로 qunatization한다
- 그리고 LoRA(Low-Rank Adapter)를 더한다.