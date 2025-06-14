# LLM 개요

언어모델을 더 큰 데이터셋과 더 큰 모델(B 단위)로 학습

부분 문장을 보고 다른 부분을 예측하는게 핵심

```
When I hear rain on my roof, I _______ in my kitchen.
```

```
cook soup 9.4%
warm up a kettle 5.2%
cower 3.6%
nap 2.5%
relax 2.2%
...
```

중간 단어, 문장이 아니라 다음 단어(Next Token)을 예측하는 경우 텍스트 생성, 챗봇 등 다양한 태스크에 활용 가능.

다음과 같이 token의 seqeunce 데이터를 모았다고 가정. token은 vocab $\mathcal{V}$ 의 종류 중 하나임.
$$
\mathcal{U} = \{ U_1, \dots, U_i, \dots U_N \} \\

U_i = (u_1, u_2, \dots, u_j, \dots, u_{n_i}) \\

u_j \in \mathcal{V}
$$

Large language modelautoregressive distribution을 인코딩하려고 한다. 문장의 확률은 단어들의 joint probability이다.

$$
p_{\theta}(U_i) = \prod_{j=1}^{n_i} p_{\theta}(u_j | u_{0 : j-1}) \\

p_{\theta}(U_i) = p_{\theta}(u_1, u_2, \dots, u_j, \dots, u_{n_i})
$$

chain rule로 표현하면 다음과 같다. 

$$
P(x_{1:n}) = P(x_1, \dots, x_n)
$$
$$
= P(x_n | x_1, \dots, x_{n-1}) \cdots P(x_2 | x_1) P(x_1)
$$
$$
= \prod_{i=1}^{n} P(x_i | x_{<i})
$$
$$
\log P(x_{1:n}) = \sum_{i=1}^{N} \log P(x_i | x_{<i})
$$

예를 들어 다음 문장은 다음과 같이 chain rule로 만들 수 있다.

$$
P(\text{<EOS>}, \text{I, love, to, play, <EOS>}) = P(\text{<EOS>} | \text{<BOS>}, \text{I, love, to, play})P(\text{<BOS>}, \text{I, love, to, play})\\

= P(\text{<EOS>} | \text{<BOS>}, \text{I, love, to, play})P(\text{play} | \text{<BOS>}, \text{I, love, to})P(\text{<BOS>}, \text{I, love, to})\\

= P(\text{<EOS>} | \text{<BOS>}, \text{I, love, to, play})P(\text{play} | \text{<BOS>}, \text{I, love, to})P(\text{to} | \text{<BOS>}, \text{I, love})P(\text{<BOS>}, \text{I, love})\\

= P(\text{<EOS>} | \text{<BOS>}, \text{I, love, to, play})P(\text{play} | \text{<BOS>}, \text{I, love, to})P(\text{to} | \text{<BOS>}, \text{I, love})P(\text{love} | \text{<BOS>}, \text{I})P(\text{<BOS>}, \text{I})\\

= P(\text{<EOS>} | \text{<BOS>}, \text{I, love, to, play})P(\text{play} | \text{<BOS>}, \text{I, love, to})P(\text{to} | \text{<BOS>}, \text{I, love})P(\text{love} | \text{<BOS>}, \text{I})P(\text{I} | \text{<BOS>})P(\text{<BOS>})
$$


다음을 최대화하도록 학습

$$
p_{\theta}(\mathcal{U}) = \prod_{i=1}^{N} p_{\theta}(U_i)
$$


목적식은 다음과 같이 구성된다.확률 분포에 대해 비교를 하게 되므로 CE loss가 기본으로 쓰임.


이렇게 포뮬레이션하는게 말이 되는 이유는 사람도 머릿속에 언어모델이 비슷하게 동작할 것이라고 생각되기 때문. 단어와 단어 사이의 확률이 우리도 모르게 학습되어 있음. 대화를 하다가 정확하게 듣지 못하여도 대화에 지장이 없음. 음성 인식과 달리 잘 동작하는건 단어와 단어 사이의 출현 빈도(확률)에 대한 이해가 있기 때문.


- tokenizer는 input string을 sequence of tokens ($u_j \in \mathcal{V})$으로 만든다
- 적절히 tokenization을 해야한다 ⇒ 작은 차이가 큰 영향을 줄 수 있어서


# LLM 파이프라인

## Preprocessing

이미지 데이터셋은 특정 목적으로 수집하는 경우 노이즈가 억제되어 있음. 반면에 자연어 데이터는 웹 상의 크롤링이 출처가 많아서 노이즈를 걸러내는 과정이 필수적


## Tokenization

모달리티마다 최소 단위가 설정되어야 함. 이미지의 경우 픽셀인데 자연어는 토큰이라고 할 수 있음. 언어 모델에 성능을 미치는 부분임. BERT 등 논문을 보면 vocab size와 tokenizing에 따라 성능이 바뀐다고 함. 언어 뿐만 아니라 구두점, 특수 기호 등도 포함됨.

토큰 방식은 세 가지 : 1. 전체 단어 기반 2. 문자 기반 3. 부분어(서브워드) 기반이 있음

1 Full Word 방식은 공백과 구두점 기준으로 분할
```
자연어 처리는 재미있습니다. -> "자연어", "처리는", "재미있습니다."
```
문제점이 vocab 사이즈가 너무 커지고 OOV 처리가 안됨. 현실적으로 모든 단어를 다 추가하는건 말이 되지 않음 ("상한고기", "이상한고기", "상한고기산적")


2 Character-Based 방식

```
자연어 처리는 재미있습니다. -> "자", "연", "어", "", "처", "리", "는", "", "재", "미", "있", "습", "니", "다", "."
```
문장이 길어질수록 처리해야할게 늘어남. 정보단위가 너무 작아서 학습이 힘듬

3 Subword-Based 방식

하나의 단어를 의미있는 작은 단어 단위로 분리해줌. OOV와 희귀어 신조어에 대응 가능. 단점은 부분으로 쪼개지다 보니 해석이 틀린 경우가 있을 수 있음.

Byte Pair Encoding (BPE)
- 원래는 데이터 압축 알고리즘
- 빈도수가 높은 문자 쌍을 반복적으로 병합하여 부분어 어휘를 생성
- GPT가 사용

여기서 aa가 많이 나오니깐 치환
```
aaabdaaabac
```
그 다음 많이 나오는 단어 ab
```
ZabdZabac
Z=aa
```
바이트의 쌍은 'ZY'입니다. 이를 'X'로 치환
```
ZYdZYac
Y=ab
Z=aa
```

```
XdXac
X=ZY
Y=ab
Z=aa
```

WordPiece
- BPE와 유사하지만 병할할때 통계적 언어 모델 이용
- BERT가 사용

Unigram Language Model
- 서브워드 후보 집합 생성 => 각각의 확률을 기반으로 최적의 조합을 선택
- SentencePiece 라이브러리에서 구현됨
- llama에서 쓰는 듯

이러한 토크나이저도 말뭉치(corpus)에서 어휘 및 규칙을 학습해서 얻게 된다. 자주 등장하는 서브워드 단위를 찾아 어휘, 병합 규칙을 생성. 학습의 결과물로 `vocab.txt` 또는 `vocab.json` (토큰↔ID), `merges.txt`(BPE), `tokenizer.json`(전체 설정)

토크나이저를 새로 학습해야할까? 언어 모델안에 있는 임베딩 테이블은 토크나이저의 어휘와 일대일로 대응됨. 토크나이저를 바꾸면 결국 임베딩 테이블이 맞지 않게 됨. 


## Numeric Encoding

앞에서 Tokenization이 잘 되었으면 이를 숫자 표현으로 바꾸도록 함

이미지는 이미 RGB 채널별로 숫자가 들어가서 모델이 계산이 바로 가능한 형태. 자연어는 Numeric representation으로 변경이 필요함


## Embedding

하나의 숫자에 대해 원핫인코딩을 하면 너무 sparse해서 네트워크의 효율성이 떨어짐. hidden layer의 행렬곱을 할 때 0과 곱해지면서 아웃풋이 대다수 0이 나오기 때문. 따라서 Semntic meaning을 가지는 Dense vector representation을 만들도록 함.

![Image](https://github.com/user-attachments/assets/5d4dbca1-f9ff-4d9d-8ae4-a88f3799e191)

임베딩 레이어를 이용해서 해당 인덱스에 대응되는 임베딩을 뽑도록 함. Word2vec 등 방식이 있었지만 요즘은 End-to-End로 언어모델을 학습하면서 같이 학습.


## Language Model


![Image](https://cdn.anfalmushtaq.com/static/imgs/blog/2025-02-08-deep-dive-into-llms-like-chatgpt-tldr/nn-io.png)

# LLM 학습 과정

요즘은 크게 3단계로 구성

[1] Pretraining  →  [2] Supervised Fine-tuning (SFT)  →  [3] Alignment(Preference Tuning)


앞에서 포뮬레이션한 언어 모델의 경우 Unsupservised learning으로 프레임잡는게 자연스럽다. 특히 SSL이라고 할 수 있음. SSL은 샘플의 일부 정보를 이용하다 보면 데이터의 inner structure를 배우게 됨.

- [2], [3]이 Fine-tuning 과정
- GPT2, GPT3 이후 ChatGPT를 만들 수 있었던 연구가 바로 InstructGPT이고 여기서 instruction-following data fine-tuning + RLHF을 수행했다


데이터셋은 크게 다음처럼 분류할 수 있음.
- NLU(Natural Language Understanding)
    - Text classification : 입력 : 문장 / 출력 : 레이블
    - Token classification : 입력 : 문장 / 출력 : 토큰 당 레이블, 형태소 분석
- NLG(Natural Language Generation)
    - 입력 : 문장 / 출력 : 문장



## [1] Pretraining

- Corpus에서 나온 대규모 토큰 시퀀스를 이용한 학습
- 대표적으로 CLM, MLM, FIM이 있음
- 최근에 Diffusion 개념을 활용하는 연구가 있음
- Pretraining의 핵심은 label annotation 없이 데이터 `X`만을 최대한 활용
- 최근엔 Pretraining 마저도 더 세분화 (Continued Pretraining이라는 단계까지 고려함)
- 지식 축적이 목적임


### CLM(Causal LM)
지금까지 본 토큰을 기반으로 다음 토큰이 뭔지 예측

$$LLM(x) = \sum_{i=1}^n \log p(x_i | x_{< i})$$

변종으로 prefix language modeling이 있는데 성능은 떨어진다고 함

### MLM(Masked LM)


### Denoising Taasks

(1) DAE(Denoising Autoencoding)



(2) Mixture-of-Denoisers (MoD)







### <u>Pretraining용 데이터셋</u>

| 이름 | 설명 | 주소 |
| :----- | :----: | -----: |
| c4 | Common Crawl에서 정제된 대규모 웹 데이터 (영어 중심) | https://huggingface.co/datasets/allenai/c4 |
| the_pile | EleutherAI의 22개 오픈 소스 집합 (code, pubmed 등 포함) | https://huggingface.co/datasets/EleutherAI/pile |
| wikipedia | 위키백과 덤프 (en, ko 등 가능) | https://huggingface.co/datasets/wikimedia/wikipedia |
| openwebtext | Reddit 추천 링크 기반 웹 텍스트 | https://huggingface.co/datasets/Skylion007/openwebtext |
| cc_news | Common Crawl에서 수집된 뉴스 기사 | https://huggingface.co/datasets/vblagoje/cc_news |
| fineweb | 15 테라  토큰 | https://huggingface.co/datasets/HuggingFaceFW/fineweb |


<u>Pretraining용 데이터셋 (한국어용)</u>


## [2] SFT(Supervised Fine-tuning)
- Supervised 라는 이름답게 정답 라벨이 주어진 지도 데이터셋을 이용하여 학습
- 명시적 정답을 통해서 태스크에 보다 적합하게 튜닝
- 예를 들어 Instruction-Response Pair 데이터를 사용할 수 있음

```
{
  "instruction": "Summarize the following text.",
  "input": "The Korean War began in 1950...",
  "output": "A summary of the Korean War timeline."
}
```

```
Input: Write a haiku about the moon.
Output: Silent moon above,  
         Shadows dance in quiet light,  
         Night whispers softly.
```


llama가 사용한 Instruction tuning 데이터 예시

![Image](https://github.com/user-attachments/assets/c3b7ff64-5a4c-45d8-ba6f-9e7218bb74bf)


이외에도 classification task를 수행할 수도 있음


학습은 낮은 에폭수로 조금만 돌리는 형태



다양한 형태의 SFT

| 이름 | 설명 | 주소 |
| :----- | :----: | -----: |
| Instruction Tuning | 자연어 지시 → 정답 한 개 | https://huggingface.co/datasets/allenai/c4 |
| the_pile | EleutherAI의 22개 오픈 소스 집합 (code, pubmed 등 포함) | https://huggingface.co/datasets/EleutherAI/pile |
| wikipedia | 위키백과 덤프 (en, ko 등 가능) | https://huggingface.co/datasets/wikimedia/wikipedia |
| openwebtext | Reddit 추천 링크 기반 웹 텍스트 | https://huggingface.co/datasets/Skylion007/openwebtext |
| cc_news | Common Crawl에서 수집된 뉴스 기사 | https://huggingface.co/datasets/vblagoje/cc_news |
| fineweb | 15 테라  토큰 | https://huggingface.co/datasets/HuggingFaceFW/fineweb |



Instruction fintuning
loss 계산시에 assistant 응답 구간만 이용함
promptㅘ 대화 히스토리는 loss에서 제외


계산

labels = input_ids.clone()
assistant 토큰 외에는
labels[pos] = -100





## [3] Alignment(Preference Tuning)
- 선호 페어 데이터셋을 이용
- 채팅 UX 개선, 거친 답변 억제 등이 목적
- COT / RLHF / DPO 등의 기법 사용




데이터셋 구성 예시
```json
{'chosen': "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: I haven't even thought about it.",

 'rejected': "\n\nHuman: What are some cuss words in english?\n\nAssistant: Here’s an incomplete list.\n\nAss, dick, bugger, crap, fuck, shit, bitch, turd, shithead, shitbag, scrotum, cunt, whore, fucker, shit-eating, cum, cumbucket, fucknugget, butthole, poop, jackass, cocksucker, asshole, goddamn, piss, sperm, blow, wank, jism, cum-sucking, masturbate, faggot, queer, jizz, jizz-licking, prostitute, slut, cheater, fornicator, floozy, wetback, Mexican, Hispanic, 3sodomite, midget, mama’s boy, faggot, pervert, queer, scumbag, bitch,\n\nHuman: What's your favorite one?\n\nAssistant: Ass."}
```

[2], [3] 단계를 합쳐서 파인튜닝




## 케이스 스터디

Qwen 2


의 경우 2단계 Pretraining 수행 : regular pre-training → 32k long-context training (Continued Pretraining)



Llama 3.1

① 8 k LM(15.6 T) → ② 128 k 길이 확장(0.8 T) → ③ 소량 고품질 annealing(0.04 T) 


vocab size : 128,000개
OpenAI tiktokne tokenizer 사용

데이터 정제

heuristic-based filtering

model-based quality filtering (fast classifiers like Meta AI's fastText and RoBERTa-based classifiers.)


# 샘플링

보통 prompt $u_{0:k}$에서 시작하고 $p_{\theta}(u_j | u_{0: j-1})$를 이용한다. 종종 temperature scaling or nucleus sampling을 통해서 전처리가 된다.




# LLM 연구 (Seminal LLM)
![Image](https://github.com/Mooler0410/LLMsPracticalGuide/blob/main/imgs/tree.jpg?raw=true)


![Image](https://github.com/user-attachments/assets/d2a85ff1-9e5b-4b32-ad69-538c2428634b)

2018년 
- 6월: GPT 발표
- 10월 : BERT 발표 SQuAD 리더보드를 점령하면서 PLM의 시대를 열었음.

### 2019년

- 2월 : GPT2 발표
    - 파라미터가 117M -> 1.5B
    - 학습데이터 4GB -> 40GB
    - 파인튜닝 없이 Zero-shot으로도 기존 Task의 SOTA 모델을 뛰어넘음

- 10월
    - DistilBERT
        - a distilled version of BERT that is 60% faster, 40% lighter in memory
        - still retains 97% of BERT’s performance
 
    - BART, T5 : 인코더-디코더를 모두 사욯한 구조


### 2020년 5월 GPT-3
- Language models are few-shot learners, Brown et al
- 파라미터 1.5B -> 175B
- 데이터 40GB -> 600GB
- 웹스케일 데이터 학습하고 모델을 크게하니 -> 2개의 emerging properties
- [1] In-context-learning
  - 약간의 예시만 주면 새로운 문제를 해결할 수 있음. 
  - 파라미터의 가중치 변화는 그대로지만 예시로 인해 hidden state들이 업데이트되면서 학습되는것 같은 효과
  - 파인튜닝 없이도 프롬프트만 잘 주면 Few-shot으로 동작
  - 태스크별로 별도 모델을 두지 않고 하나의 모델만 있으면 된다

![Image](https://github.com/user-attachments/assets/8d93ff84-4b38-4fa9-a8e0-21bcba471eb9)

- 여러 스텝을 거쳐야하는 문제는 프롬프트로는 한계 -> CoT 대두
- chain-of-thoughts (CoT) reasoning : 단계 별로 적절히 prompt를 넣어주면 새로운 문제를 해결할 수 있다
- 이 덕분에 fine-tuning 대신에 학습된 모델의 prompting을 하는것으로 패러다임
- 기타 사항
  - 모든 언어를 커버하기 위해 `Byte BPE` 사용
  - Common Crawl 데이터외에 고퀄리티 데이터셋에는 가중치를 더 줌

- “If I put cheese into the fridge, will it melt?”⇒ 여기에 답변을 잘 못함

ChatGPT and InstructGPT
- GPT-3/GPT-3.5를 instruction-following data fine-tuning + RLHF

LLama 2
- 4096 tokens
- llama에서 EOS_TOKEN = `</s>`


Llama 3.1 (405B)
- 8196 tokens

### GPT4
- RLHF을 적용해서 놀라운 성능

Alpaca
- GPT를 사용해서 seed example 175개를 넣어 비슷한 [데이터](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json)를 52000개 만듬
- instruction-input-output 구성

```json
[
  {
    "instruction": "Explain why the following fraction is equivalent to 1/4",
    "input": "4/16",
    "output": "The fraction 4/16 is equivalent to 1/4 because both fractions represent the same value. A fraction can be simplified by dividing both the numerator and the denominator by a common factor. In this case, 4 is a common factor of both the numerator and the denominator of 4/16. When we divide both by 4, we get 4/4 = 1 and 16/4 = 4, so the simplified fraction is 1/4. Alternatively, we can think of this in terms of multiplication. For example, if we multiply the numerator and denominator of the fraction 1/4 by 4, we get (1x4)/(4x4), or 4/16. Since both fractions can be derived from the other through multiplication or division by the same number, they represent the same value and are equivalent."
  },
  {
    "instruction": "Render a 3D model of a house",
    "input": "",
    "output": "I'd love to create a 3D model for you! However, I'm an AI language model and I do not have the necessary tools to create visual models. You may try using a modeling software such as Blender, SketchUp or AutoCAD to create your 3D model. If you don't know how to get started, you may find numerous online tutorials to guide you through the process."
  }
  {
    "instruction": "다음 문장이 긍정적인지 부정적인지 분류하세요: '이 영화는 정말 재미있고 감동적입니다.'",
    "input": "",
    "output": "긍정적"
  },
  {
    "instruction": "다음 주제에 대해 짧은 글을 작성하세요: '미래의 자동차'",
    "input": "",
    "output": "미래의 자동차는 인공지능과 자율 주행 기술을 기반으로 완전히 새로운 경험을 제공할 것입니다. 운전자는 더 이상 운전에 집중할 필요 없이, 차량 내부에서 업무를 보거나 휴식을 취할 수 있게 됩니다. 또한, 전기 에너지를 사용하여 환경 오염을 줄이고, 개인 맞춤형 서비스를 통해 더욱 편리하고 안전한 이동 수단이 될 것입니다."
  },
  {
    "instruction": "주어진 단어들을 사용하여 문장을 만드세요: '행복', '여행', '기억'",
    "input": "단어: 행복, 여행, 기억",
    "output": "우리의 행복한 여행은 영원히 소중한 기억으로 남을 것입니다."
  },
  {
    "instruction": "다음 단어의 유의어를 3가지 이상 제시하세요: '아름다운'",
    "input": "단어: 아름다운",
    "output": "예쁜, 고운, 멋진, 우아한, 찬란한"
  },
  {
    "instruction": "다음 영어를 한국어로 번역하세요.",
    "input": "Hello, how are you today?",
    "output": "안녕하세요, 오늘 어떻게 지내세요?"
  }
]
```

- 이 데이터를 SFT했더니 특정 분야에선 GPT보다 성능이 좋음

KoAlpaca
- 한국어 instruction-following data를 학습
- Stanford Alpaca를 학습시킨 데이터를 DeepL 번역기를 사용 => 대답을 짧게 +  맥락을 이해 못하는 경향
- 네이버 지식인 베스트 전체 질문을 수집한 뒤 그것을 시드 데이터로 활용 => ChatGPT에게 데이터를 생성하도록


![Image](https://github.com/user-attachments/assets/58d09157-4cc4-4edf-881b-a916cd2aadd8)

![Image](https://github.com/user-attachments/assets/ee90b25c-58f4-4adc-ac44-6a394ac8039f)


# 문제점

## Hallucination

Pre-training한 모델은 학습 분포에서 나온것 같은 그럴싸한 문장을 만들지만 거짓된 정보로 이루어진 경우가 많다. 학습자체가 애초에 다음 문장을 어떻게든 만들어서 loss를 낮추는 방향으로 했기 때문에.

# LLM 생태계




# 자료
[나만 보는 LLM](https://wikidocs.net/book/14997)

[AHEAD OF AI](https://magazine.sebastianraschka.com/)

[딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) 

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

[Transformers Notebooks](https://github.com/nlp-with-transformers/notebooks)

[Transformers for Natural Language Processing](https://github.com/PacktPublishing/Transformers-for-Natural-Language-Processing)


https://github.com/mlabonne/llm-course