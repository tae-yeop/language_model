# BERT

### 개요

Bidirectional Encoder Representations from Transformers
Transformer Encoder만을 활용하여 각 토큰을 처리

두 가지 형태의 사전 학습 수행

1) MLM
- 일정 비율의 토큰을 가린채, 문장을 복원하도록 학습
- 랜덤하게 15프로 선택
- 이 중 80프로 [MASK]로 변경
- 10프로는 랜덤한 단어로 변경 : `i love to go to school ⇒ i love to go to jail`
- 10프로는 동일하게 놓아둠 (일부러 헷갈리게 해서 언어모델을 고도화)
- 이런 15프로에 대해 제대로된 단어를 예측해야만 함


![Imgae](https://wikidocs.net/images/page/115055/%EC%A0%84%EC%B2%B4%EB%8B%A8%EC%96%B4.PNG)



- `dog` 토큰을 [MASK]로 변경하고 이를 맞춰야 함
- loss 계산시 다른 토큰은 무시하고 output을 내놓는 토큰(Missing word token)만 계산에 참여


![Image](https://wikidocs.net/images/page/115055/%EA%B7%B8%EB%A6%BC8.PNG)

- 3개의 변경 사항이 모두 적용되어도 이 3개를 모두 똑바로 예측해야함

![Image](https://wikidocs.net/images/page/115055/%EA%B7%B8%EB%A6%BC9.PNG)


- 15프로만 하는 이유는 학습과 추론의 괴리를 없애기 위해서 (파인튜닝이나 추론시에는 [MASK]가 없는 인풋을 제공하게 됨)
- Next step token prediction이 아님 => 방향성과 자기회귀 성질이 없어짐 => 양방향 언어모델(Bi-directional LM)이 되어질 것임 => 문장 전체를 봐야하는 NLU에서 좋은 성능

-  NLU Task에서 PLM의 성능을 대폭 개선


2) Next Sentence Prediction

![Image](https://github.com/user-attachments/assets/71c2c8c0-3bc7-40f0-8a96-b7c7cb66a573)

- Question Answering 또는 Textual Entailment task의 경우 문장 사이의 관계를 이해하는 것이 중요함 (감정 분석은 문장 내부를 이해하면 끝인 반면)

- 두 문장간의 관계를 학습하길 기대함


```
(Input = [CLS] That's [mask] she [mask]. [SEP] Haha, nice ! [SEP] , Label = IsNext)

(Input = [CLS] That's [mask] she [mask]. [SEP] Dwight, you ignorant [mask] ! [SEP] , Label = NotNext)
```
- [SEP] (separator)으로 분리되는 두 문서 (A,B)를 통과 시킴

- B를 50% 확률로 임의의 문서로 대체 ⇒ 만약 연결되지 않은 문서인 경우 ⇒ <NotNext>
- 50% 확률로 연결된 문서를 선택할 수도 있음 ⇒ <IsNext>
- <CLS> (class) 스폐셜 토큰의 위치에서 대체 여부 예측하도록 학습 ⇒ 이 위치에서 <NotNext>인지 <IsNext>인지 결정함
- 나중에 finetuning할 때 <CLS> 토큰에 Linear layer를 붙히고 softmax를 붙혀서 사용하기도 한다
- 따라서 <CLS> 토큰은 맨 앞에 붙어서 굉장히 중요함


이렇게 사전학습한 모델에 Head 레이어를 달아서 다운스트림 태스크를 수행하도록 파인튜닝할 수 있었음. Natural Language Understanding에는 좋으나 Natural Language Generation에는 별로임. 이해가 필요한 번역, 요약 태스크에 좋은 성능을 발휘함. 



### BERT에서 임베딩

![Image](https://github.com/user-attachments/assets/4f15fbf2-dcf1-4c25-9c67-dc0aafcfde3a)

BERT에선 Segment embedding을 추가해서 문장을 구분할 수 있도록 함

### 어텐션 마스크

![Image](https://wikidocs.net/images/page/115055/%EA%B7%B8%EB%A6%BC11.PNG)

[PAD] 토큰에는 어텐션이 걸리지 않게 하는 역할


### BERT 파인튜닝 활용

Classification과 QA용 태스크에 활용가능
![Image](https://github.com/user-attachments/assets/5c60189e-c26a-45d7-ab39-ee96758c3fd7)


Spanning : Weight vector $S$와 $E$ 추가
![Image](https://github.com/user-attachments/assets/1ad62c43-308a-4708-b267-faf7616e8dbd)

- 문장의 시작과 끝을 알려준다
- 예를 들어 이순신 장군에 대한 위키 페이지를 입력에 넣고 언제 전사했는지 물어보면 이와 관련된 문장의 시작과 끝을 알려주면 된다
- $s_i, e_i$ : 시작 인덱스, 끝 인덱스, $x_i$ : 정답이 들어있는 문서
- $S,E \in \Bbb{R}^h$ : hidden size의 벡터
- 모든 timestep $j=1,.. l$ 까지 다 고려해서 분모로 ⇒ softmax를 만듬
- 전체 timestep 대비 i번째가 시작이 되는 확률


### 한국어 BERT


# RoBERTa

## 설명

RoBERTa(Robustly Optimized BERT)는 BERT를 좀 더 잘 깍아서 더 성능을 높인 연구

기존의 BERT에 더 많은 데이터와 더 나은 hyper parameter을 통해 BERT 이후 PLM보다 더 나은 성능을 얻을 수 있음을 보임 ⇒ BERT가 아직도 underfitting

BERT 이후 다양한 변종이 나왔지만 BERT만 잘 주무르면 괜찮은 성능을 낸다는 것을 보여줌

### 개선 사항

1) Training the model longer, with bigger batches, over more data

- 기존 BERT 대비 10배 가량의 데이터를 더 오래 학습

2) Removing the next sentence prediction objective

- 실험을 통해 NSP가 불필요함을 보임

3) Training on longer sequences

- 최대 입력 길이(512)에 맞춰 최대한 문장을 인풋으로 넣음

4) Dynamically changing the masking pattern applied to the training data

- Feed-forward 때마다 다이나믹하게 masking을 새롭게 구성
- batch를 뽑아왔을 때 마스킹 처리
- 기존 BERT에선 특정 단어에 <mask>를 처리함 ⇒ 해당 단어는 모든 문장에서 <mask> 처리됨 (학습 이전에 한번 처리하고 끝까지)


### 결론

BERT 이후로 나온 XLNet은 복잡한데 BERT만 잘 만져도 좋은 성능을 얻을 수 있음. NSP 방식을 안써도 성능이 좋았다. 




# CLM 계열

## GPT



![Image](https://github.com/user-attachments/assets/21568391-011d-4ccf-9293-c003ad4f22bf)

Transformer Decoder만을 활용하여 언어 모델 사전학습. 단순한 언어모델의 objective로 큰 모델을 학습하고 전이학습에 활용해보니 아주 효과적(적은 수의 labeled dataset으로도 Supervised learning과 비슷한 성능)

이때는 Task별로 전이학습을 수행해서 - 입력 : 텍스트와 특수 토큰을 결합 ⇒ 하나의 시퀀스를 만듬
- 출력 : 마지막에 layer를 추가하여 $\hat{y}$을 얻음

![Image](https://github.com/user-attachments/assets/9037187a-35e4-4306-a362-7f4acbb0679c)