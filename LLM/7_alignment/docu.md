

RLHF



DPO training은 데이터 수집 후 최적화 작업

1. 데이터 수집 : 프롬프트가 주어지면 긍정적이고 부정적인 세대 쌍으로 선택된 선호도 데이터 세트를 수집
2. 최적화 : DPO 손실의 로그 가능도를 직접 최대화

DPO 장점
reward model 학습과 PPO 학습 단계 2개를 건너띄고 preference annotated data에 대해 직접적으로 학습


Trasnformers 라이브러리 DPO Trainer 지표

Eval_samples_per_second: 모델이 처리하는 샘플 속도
Loss: 훈련 데이터에 대한 모델의 손실
Eval_loss: 평가 데이터셋에 대한 모델의 손실
Logits/Chosen: 모델이 선택된 응답에 대해 계산한 로짓 값, Logits/Rejected보다 커야지 모델이 선호되는 응답을 더 높게 평가하고 있는 것이며, 잘 학습되고 있는 것
Logits/Rejected: 모델이 거부된 응답에 대해 계산한 로짓 값
Rewards/Chosen: 모델이 선택된 응답에 대한 보상 값. 보상 값은 모델이 인간의 선호도를 얼마나 잘 학습하고 있는지를 나타냄. Rewards/Rejected보다 높은 값
Rewards/Rejected: 모델이 거부된 응답에 대한 보상 값. Rewards/Chosen보다 낮은 값
Rewards/Margins: Rewards/Chosen과 Rewards/Rejected 간의 차이
Rewards/Accuracies: 모델이 인간의 선호도와 일치하는 응답을 선택한 비율로, 정확도
Logps/Chosen: 선택된 응답의 로그 확률 값의 합.  Logps/Rejected보다 커야됨.
Logps/Rejected: 거부된 응답의 로그 확률 값의 합



# ORPO

상대적으로 적은 데이터와 메모리를 사용
데이터셋은 DPO와 같은 prompt, chosen, rejected 세가지의 컬럼
