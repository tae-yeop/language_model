모델 전체 weight를 학습하는게 아니라 적은 파라미터만 학습




QLoRA

파인튜닝할 때 두루두루 적용할 수 있다

### QLoRA

- 효율적인 파인튜닝 테크닉
- 사전에 학습한 LLM을 4bit로 qunatization한다
- 그리고 LoRA(Low-Rank Adapter)를 더한다.