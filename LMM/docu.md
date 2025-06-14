멀티모달리티를 다룰 수 있는 모델

Large Multi-modal Model or Multimodal Large Language Models

LLM의 텍스트 임베딩 스페이스로 매핑하거나 공유되는 하나의 임베딩 스페이스로 매핑함
LLM이 두뇌가 되어서 다른 모달리티를 인식하는 식으로 구성
이를 위해 Projection Module이 필요함

- Flamingo : Frozen image encoder — CA — LLM
- BLIP-2 : image — Q-former — LLM
- LLaVA : image features — simple projection scheme — word embedding space`


![Image](https://github.com/user-attachments/assets/9e472404-e64d-4ba2-8dc1-f8e8b4be2266)

비전 인코더의 피처를 이용함


LLM 쪽으로 간 토큰은 기존의 GPT 모델의 Next token prediction을 그대로 수행함. 대신에 이미지 토큰에 대해선 Self-Attention을 수행하는 식으로 하고 텍스트 토큰에 대해선 Casaul Attention Mask를 적용한다.

![Image](https://github.com/user-attachments/assets/5fcd3674-348c-4fd2-94ca-365cf628dcfb)



케이스 스터디

대부분 모델은 다량의 image-text pair로 학습. 학습 전략은 연구마다 다르다. BLIP-2의 경우는 이미지 인코더와 LLM은 고정하고 Q-former를 학습시킴. 

![Image](https://github.com/user-attachments/assets/614d054a-697d-41e6-9059-28b9297088a5)

Flamingo는 사전에 학습한 이미지 인코더와 언어 모델(친칠라)에 Perciver Sampler를 사용함. Perceiver Sampler의 피처를 LM에 Gated Cross Attention 모듈이 받아서 모달리티 정보를 통합.

![Image](https://github.com/user-attachments/assets/3d7c7819-0ac7-4c53-9d47-8d02bc8f719f)

Flamingo, BLIP2 같은 모델에서  Mutlimodal In-Context learning를 기대해볼 수 있다

![Image](https://github.com/user-attachments/assets/f6fae328-4398-4d72-8216-0696e6270614)

GPT4 부턴 이미지도 이해할 수 있어서 멀티모달모델임.

![Image](https://github.com/user-attachments/assets/2ce7a44d-1c2b-4319-83b1-f6267805df4b)



예시
- 입력으로 아예 처음부터 bbox같은 grouding 정보를 다 넣을 수도 있음


![Image](https://github.com/user-attachments/assets/97a041ad-b4c4-4663-9141-528ac9955624)

![Image](https://github.com/user-attachments/assets/1c0fddca-b950-4df9-82b9-39f694f824f3)


학습은 다양하게 할 수 있는데 예시

LLava는 다음과 같이 학습함


- Stage1 : Pre-training for Feature Alignment
    - CC3M 데이터셋 일부를 이용해서 projection matrix만 학습시킨다.
    - 두 개의 modality를 align시키는게 목적
    - Projection matrix는 이미지 토큰을 Language model이 이해할 수 있는 형태로 변화시킨다
- Stage2 : Fine-tuning End-to-End
    - Projection과 LLM을 같이 학습시킨다
    - 만들어 놓은 Instruction 데이터를 이용


관련 데이터셋
MMC4(Multimodal C4 dataset)
MIMIC-IT :  2.4M multimodal instruction 데이터셋, 2개 이미지와 답변을 ICE로 주었을 때 이 답변의 스타일로 대답
보통 VQA 데이터셋

Image Captioning
Visual Reasoning



# Any-to-Any Multi-Modal

이미지외에 오디오, 스피치, 비디오 등도 처리할 수 있게 발전된 모델들

![Image](https://github.com/user-attachments/assets/9b9924df-b090-405b-a58e-dfc85ffb9e3b)

Next-GPT는 multimoda adapter와 diffusion decoder를 LLM에 연결. ImageBind 사용하여 6개 모달리티에 대응되는 임베딩 스페이스를 사용. 각 모달리티별 별도의 토큰을 사용. 


![Image](https://github.com/user-attachments/assets/a65b0177-37ec-413e-baae-d66fa1e69913)

학습은 단계별로 수행하는데 1단계는 LLM이 인코더의 출력을 처리할 수 있게 함. 

![Image](https://github.com/user-attachments/assets/acb96848-3770-4d64-a952-da0a34a10958)


2단계에선 LLM 아웃풋 instruction과 diffusiodn model을 일치시킴. 원래 Diffusion 모델이 받아들이는 text token과 LLM에서 나오는 token과 괴리를 메꾸도록 함. Diffusion 모델은 고정

![Image](https://github.com/user-attachments/assets/4c4a820c-8056-494c-b18e-8e6c74ecf2b4)

3단계는 Modality-switching Instruction Tuning을 수행하여 User Instruction에 맞는 아웃풋을 내도록 학습함. 
![Image](https://github.com/user-attachments/assets/1ee0ffb6-3117-4273-aba4-bf2ff19c0888)


## 관련 데이터셋

![Image](https://github.com/user-attachments/assets/4f3fc2eb-83ca-4686-b9db-beafd4e31f14)