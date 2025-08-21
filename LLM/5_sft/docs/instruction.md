## Supervised Learning

여기서 핵심은 입력 X를 주고 의도에 맞는 Y를 출력하도록 학습하는 것. 즉 정답 라벨을 추가해서 학습

### 토큰을 추가해서 학습하기

추가적인 토큰들이 들어가서 학습되어야 하는 경우가 있다. 이 경우 tokenizer와 model에서 살짝 조정이 필요하다. 스폐셜 토큰의 경우는 `add_special_tokens()`을 써야 하고 일반 vocab은 `add_tokens()`을 쓰도록 하자. 

```python
# 예시: 새로운 bos_token과 eos_token을 추가
tokenizer.add_special_tokens({'bos_token': '[BOS]', 'eos_token': '[EOS]'})

# 또는 단순히 새로운 특수 토큰을 추가
tokenizer.add_special_tokens({'new_special_token': '<MY_SPECIAL_TOKEN_FOR_TASK>'})
```

추가적인 vocab으로 인해 임베딩이 바뀌어야 하므로 임베딩 레이어를 조절하는 메소드가 있음. 


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

print(f"변경 전 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")
print(f"변경 전 토크나이저의 어휘 크기: {len(tokenizer)}")

# 새로운 토큰 추가
new_tokens = ["<special_word>", "<new_concept>"]
num_added_tokens = tokenizer.add_tokens(new_tokens)
print(f"추가된 토큰 개수: {num_added_tokens}")
print(f"변경 후 토크나이저의 어휘 크기: {len(tokenizer)}")

# 모델의 임베딩 레이어 크기 조절
# 새로 추가된 토큰에 대해 무작위로 초기화된 임베딩 벡터가 생성됩니다.
model.resize_token_embeddings(len(tokenizer))
print(f"변경 후 모델의 임베딩 크기: {model.get_input_embeddings().weight.shape[0]}")

# 이제 새로운 토큰을 사용하여 텍스트를 인코딩하고 모델에 전달할 수 있습니다.
# 주의: 새로 추가된 토큰의 임베딩은 무작위로 초기화되므로,
# 이 토큰이 의미 있는 역할을 하려면 추가적인 미세 조정(fine-tuning)이 필요합니다.

```
 
## Instruction Tuning

대표적인 SFT 방식은 언어 모델에게 Task Instruction를 주어서 이에 따르는 응답을 내놓도록 하는 SFT의 일종이다. 

### 역사적 맥락

2020년 GPT3는 `In-Context Learning(ICL)` 개념을 보여줌. 이는 프롬프트에 컨텍스트 예시를 몇개 주고 퀴즈를 내었을 때 맞추도록 했더니 잘 수행하는걸 보여줌

![Image](https://ludwig.ai/latest/images/icl_few_shot_learning.png) 

이를 통해 Prompt를 잘 넣으면 다양한 태스크를 수행할 수 있다는 사실을 알게 되어 `Prompt Engineering`이 유행하게 됨. 이는 모델 파라미터는 고정이고 대신 인풋으로 들어간 예시를 보고 새로운 태스크를 수행하도록 내부적으로 조정된다는 것을 뜻함. 어차피 모든 딥러닝 모델은 Linear+Activation으로 환원할 수 있고 인풋 $X$과 파라미터 $W$간의 matmul 결과가 다음에 전달됨. 이때 $W$는 고정이더라도 인풋이 다르면 $X \cdot W^{T}$의 결과가 달라지는 효과가 있음.

그런데 모든 작업에 일관된 성능을 보장하기 어렵고 `Prompt Engineering`은 번거롭고 수고스럽다. 그냥 아예 직접 파인튜닝 하는게 속 편하다. 예를 들어 번역 작업의 경우 Input (영어)와 Output(중국어)로 대응되거나 요약 작업의 경우 Input(원본 문장)과 Output(요약문)으로 페어를 구성하는 Supervised learning으로 파인튜닝. 이때 인풋과 타켓간의 정확한 관계는 데이터상에는 명시적으로 주어지진 않은걸 볼 수 있음. 

![Image](https://github.com/user-attachments/assets/453c7009-3126-4ecc-9edc-a59493c0fa15)

이때 ICL에서 Instruction을 줬듯이 Instruction과 이에 대한 정답 Response를 줘서 학습하는 Instruction Tuning이 힘을 얻기 시작. 지시를 통해 명확한 관계가 명시되는 Instruction-based Dataset. 2021년에 나온 구글의 FLAN(Finetuned Language Models are Zero-Shot Learners, Jason Wei et al.,) 2022년의 OpenAI InstrcutGPT로 인해 유행하기 시작.

![Image](https://github.com/user-attachments/assets/f28800a5-1684-4f68-acf7-1d8aa8695097)

이렇게 했더니 학습한 태스크가 조합된 새로운 태스크도 Zero-shot으로 수행 가능함 (사람과 비슷)

![Image](https://github.com/user-attachments/assets/2e35f9c1-00c4-4182-af1e-b98cc3e03d63)


Instruction Tuning Dataset은 Instruction(시스템설명) + (Context) + Answer로 구성됨. 모델은 Answer를 정답으로 내놓도록 학습함. Context를 Input이라고 부르기도 하고 Answer을 Response, Output라고 부르기도 함. 결국 챗봇 형태로 지시에 대해 어떤 기능을 수행하여 답을 내놓는 챗봇 형태로 LLM이 사용될 수 있게 됨.

### 데이터 예시

대표적으로 다음과 같은 message conversation 구조를 많이 쓰는 것으로 보임. Instruction Tuning dataset 뿐만 아니라 일반 채팅도 모두 쓸 수 있는 구조라고 보면 될 듯. 원래 채팅 데이터는 사람이 손으로 만들었는데 이제 UltraChat 같은걸 이용해서 생성 데이터를 만들어서 쓰기도 한다.
```
"messages": [
    {"role": "user", "content": "안녕, 오늘 날씨 어때?"},
    {"role": "assistant", "content": "안녕하세요! 서울은 맑고 25도입니다."},
    {"role": "user", "content": "내일은?"},
    {"role": "assistant", "content": "내일은 구름 많고 23도 예상됩니다."}
]
```
이는 `dict of list` 이기 때문에 멀티턴을 지원할 수 있음. "role": "user", "role": "assistant", "role": "system"을 명시해서 대화 참여자의 역할을 명확히 구분해둠. 오픈 소스 모델마다 차이점이 있다면 role의 종류가 다를 수 있음 : `user`, `assistant`, `human`, `bot`, `system`. 

`system`은 전반적인 지침을 명시함. 이를 넣는건 옵션이긴함. 만약 학습할 때 `system`을 넣지 않은 데이터로 학습했다면 추론할 때에도 `system`을 넣지 않은 채로 넣어줘야함. 꼭 있어야 하는건 `user`와 `assistant`임. 드물게 `id`, `name`, `tool_calls` 같은 함수호출이나 특정 기능을 위한 필드가 있을 수 있음

자연어 프롬프트 상에선 저런 형식을 쓰더라도 토큰화되면 모델마다 확실히 달라짐. `tokenizer.apply_chat_template()`가 어떻게 바꾸는지 규칙은 `chat_template.jinja` 파일에 명시되어 있음. 이 메소드를 쓰면 특수 토큰이 적용되면서 하나의 긴 문자열이 나오는데 이를 `chat templates`이라고 부름. 허깅페이스 토큰나이저를 통해 어떤 템플릿인지 알 수 있음.

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)

print(tokenizer.default_chat_template)

>>>
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% elif true == true and not '<<SYS>>' in messages[0]['content'] %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.' %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\n' + content.strip() + '\n<</SYS>>\n\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
```


Llama2의 경우 다음처럼 바뀜

```
# messages = [{"role": "user", "content": "Hello!"}]
# -> <s>[INST] Hello! [/INST]

# messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]
# -> <s>[INST] <<SYS>>\nYou are helpful.\n<<SYS>>\n\nHello! [/INST]
```

Mistral / Mixtral (Mistral AI)의 경우


```
# messages = [{"role": "user", "content": "Hello!"}]
# -> <s>[INST] Hello! [/INST]

# messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]
# -> <s>[INST] You are helpful.\nHello! [/INST]
```

Gemma (Google)

```
# messages = [{"role": "user", "content": "Hello!"}]
# -> <bos><start_of_turn>user\nHello!<end_of_turn>\n<start_of_turn>model\n

# messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]
# -> <bos><start_of_turn>user\nYou are helpful.\nHello!<end_of_turn>\n<start_of_turn>model\n
```

그런데 초기에는 저런 message conversation 구조를 쓰기보단 직접적으로 Instruction, Context, Answer를 명시한 포맷 구조를 사용함. 이 또한 모델마다 연구마다 포메팅이 다름.

```
{
    Instruction
    Explain the concept of a bubble sort algorithm to a non-technical audience.

    Answer
    A bubble sort algorithm is a type of sorting algorithm that is used to sort elements in an array. It works by looking at each element of the array and comparing it to the next element. If the first element is bigger than the second element, they are swapped. This process is repeated until the whole array is sorted. This type of sorting is one of the simplest sorting algorithms, but it can be slow if the array has many elements.
},
{
    Instruction
    Make the second sentence shorter.

    Context
    Winter is usually the coldest season of the year. Snow is a common element during winter.

    Answer
    Winter is the coldest season, often accompanied by snow.
}
```

[tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) 데이터셋 예시

```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Classify the following into animals, plants, and minerals

### Input:
Oak tree, copper ore, elephant

### Response:
Oak tree: Plant
Copper ore: Mineral
Elephant: Animal
```


---

정리하면 Instruction Tuning을 하려면 앞에 두 단계가 추가되었다고 볼 수 있다 :  “Converting train dataset to ChatML" => “Applying chat template” => “Tokenizing” => “Truncating”


### Self-Instruction

그런데 이런 Instuction Dataset을 만드는데 비용이 많이 든다.  Self-Instruct: Aligning LM with Self Generated Instructions에서 제안한 방법. GPT4를 활용하도록 하는데 몇 개 예제를 만들어 놓어서 context로 넣어서 더 많은 Instruction과 Response를 얻는다. 이후 이를 다시 이를 인풋으로 넣어서 더 많이 불린다?

![Image](https://github.com/user-attachments/assets/1a0825d3-950b-4316-8fca-50d6eb9f468b)