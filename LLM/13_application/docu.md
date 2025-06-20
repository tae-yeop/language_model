LLM 모델을 활용한 다양한 Application


### LLMTime
- https://github.com/ngruver/llmtime
- https://ar5iv.labs.arxiv.org/html/2310.07820


- 왜 성능이 좋은가? LLM의 능력 덕분이라 생각
    - multimodal distributions을 자연스럽게 표현 능력
    - 여기에 biases for simplicity, and repetition ⇒  이는 seasonality와 같은 특징과 맞아 떨어진다
- non-numerical text만으로도 imputation이 가능
- GPT-4가 더 성능이 안 좋았다 (아마 RLHF같은 추가 과정으로 인해 숫자를 tokenization하는 방식 + poor uncertainty calibration 때문일 듯)

시계열 또한 언어 모델링과 유사하다 : $u_j$가 대신에 numerical

$$
\{ U_i = (u_1, \dots, u_j, \dots, u_{n_i})\}
$$

- 언어모델은 시계열 숫자를 tokenization 잘 못함
    - BPE가 학습 데이터내의 빈도수로 압축하기 때문 ⇒ 이상한 숫자로 chunk가 묶임
- LLAMA 같은 경우는 디폴트로 개별적인 숫자로 잘처리 해줌

숫자를 텍스트로 인코딩 ⇒ text completion으로 가능한 샘플링 extrapolation 수행

- 타겟 데이터셋에 파인튠하지 않고 zero-shot으로 동작
- llama는 tokenizer를 만들 때 개별적인 digits를 매핑하도록 함 ⇒ 수학 능력이 GPT-4보다 더 뛰어남

- 언어 모델에 시계열 데이터 적용한거 평가하기
    - MAE보다는 CRPS 사용
    - CRPS는 데이터의 핵심 구조를 무시함(시간 사이의 correlation)
    - 다행히 언어 모델은 전체 sequence에 likelihoods를 부여할 수 잇다

- PromptCast
    - prompt를 사용해서 QA 문제로 forecasting 문제를 해결
- 여기선 숫자 데이터 자체를 preproecssing을 엄청 잘하기만 하면 된다는 것을 보여줌
- 성능 자체는 LLM이 extrapolation을 얼마나 잘하는지에 달려 있음 (영어나 언어의 문제가 아님)