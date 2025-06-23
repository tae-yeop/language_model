
# 개요

- Isotropic Architecture 사용이 대세
- 블록의 인풋과 아웃풋 텐서 shape가 같다
- 따라서 디자인 스페이스를 보면 바뀔 수 있는 곳은 다음과 같다.

(1) Positional Embedding

(2) PreNorm

(3) Attention 

(4) FeedForward 

(5) Activatioon


# Positional Embedding

이는 Attention이 걸려서 단어의 순서 정보가 

# PreNorm

## LayerNorm
- 

## RMSNorm

## SwiGLU

# Attentions

Causal Self attentions

Gourp Qeury Attention




# FeedForward


Mixture-of-Experts



# Encoder-Decoder 모델

최초에 제안된 Transformer가 제안된 형태는 다음과 같이 기존의 Seq2Seq 모델의 논리를 따라서 설계함. Encoder stack의 마지막 블록이 내놓는 임베딩이 의미를 잘 축약하고 효과적으로 담고 있다고 가정함. Decoder에선 이 임베딩을 받아 같은 의미를 내놓는 어떤 무언가를 내놓도록 설계함.

![Image](https://jalammar.github.io/images/t/The_transformer_encoder_decoder_stack.png)

이때 Decoder는 Self-Attention 
그런데 이 구조는 언어 모델 계열에선 그렇게 유행하진 않고 있음. 이 부분에 더 exploration하는지는 아직 살펴보지 않았음.