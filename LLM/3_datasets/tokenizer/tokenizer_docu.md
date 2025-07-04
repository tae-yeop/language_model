SentencePiece에는 BPE, Unigram이 구현되어 있음


### 토크나이저 문제점

아무리 잘 자른다고 해도 언어의 특성을 모두 제대로 반영한 토크나이제이션이 된다고 보장을 못함. 결국 알고리즘적으로 스플릿한 것일 뿐. stochastic parrot이라는 비판을 받는 이유의 근원 중 하나일듯.

```
how many Rs are in the word "strawberry"?
```

![Image](https://github.com/user-attachments/assets/0f095b58-590b-408c-a4aa-907eeb255347)

https://tiktokenizer.vercel.app/에서 해보면 다음처럼 토큰이 나오게 된다.

![Image](https://github.com/user-attachments/assets/671e5613-d746-4a20-8062-bdc2a13b5bb7)


간단한 수학관련 질문을 잘 못하는 경우가 있음. 이 또한 토크나이제이션의 문제에 기인하는 듯. 좀 더 복잡한 수학 계산은 아예 코드를 돌려서 계산해서 정확한 값을 얻음. 

![Image](https://github.com/user-attachments/assets/16501b4f-2675-46f1-a10f-eb98f3dcfbe8)


토큰을 보면 11을 한 단위로 토큰화했음.

![Image](https://github.com/user-attachments/assets/2e8e1420-ad92-4973-b289-5319db09854b)

반면에 제대로된 답변을 하는 gemini의 경우 11을 따로 나눠서 토큰화했음. 

![Image](https://github.com/user-attachments/assets/3d9d7b63-c35c-46e2-aafd-0883ef1eaafe)

![Image](https://github.com/user-attachments/assets/23fe44f5-45f7-4801-a243-ece32f6fe8c3)




같은 의미인데 살짝 다르다고 토큰값이 다르게 나온다. gemma3 토크나이저를 써서 해보면 다르게 나옴.
```
tokenizer.encode(' hello'), tokenizer.encode('hello')
>>>
([2, 29104], [2, 23391])
```


이렇게 토크나이저에 의존하지 않는 연구들이 나오고 있다.

- BLT
    - BLT는 엔트로피 기반으로 계산된 패치를 이용함 ⇒ fixed vocab이 아님
    
![Image](https://github.com/user-attachments/assets/45f34da9-2893-4434-86d0-e5bf73deb743)


From Bytes to Ideas: Language Modeling with Autoregressive U-Nets
- BPE를 쓰지 않으려고 하는 또 다른 아이디어