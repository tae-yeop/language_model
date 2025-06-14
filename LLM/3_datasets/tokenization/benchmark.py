import time
import nltk
import spacy
from transformers import BertTokenizer, GPT2TokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import re
import tiktoken

kor_text = """자연어 처리는 인공지능 분야 중 하나로, 인간이 사용하는 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술입니다. 토크나이저는 이러한 자연어 처리에서 매우 중요한 역할을 합니다. 토큰화는 문장을 단어, 형태소 또는 서브워드 등으로 분할하는 과정으로, 이후의 처리 단계를 위해 텍스트 데이터를 구조화된 형태로 변환합니다.

예를 들어, 형태소 기반 토크나이저는 한국어와 같은 교착어에서 어간과 어미를 분리하여 보다 정확한 분석을 가능하게 합니다. 반면에 서브워드 토크나이저는 희귀 단어나 미등록 단어를 처리하는 데 효과적입니다. 이러한 토크나이저의 선택은 모델의 성능과 효율성에 직접적인 영향을 미칩니다.

최근에는 딥러닝의 발전으로 인해 BERT, GPT와 같은 대형 언어 모델이 등장하였으며, 이들은 주로 서브워드 토크나이저를 사용합니다. 이러한 모델들은 대량의 데이터를 기반으로 사전 학습되어 다양한 자연어 처리 태스크에서 뛰어난 성능을 보여줍니다.

토크나이저의 성능을 비교하기 위해서는 동일한 텍스트를 다양한 토크나이저로 처리하여 작업 속도와 토큰화 결과를 분석해야 합니다. 이를 통해 각 토크나이저의 장단점을 파악하고, 특정 응용 분야에 가장 적합한 토크나이저를 선택할 수 있습니다.

이 예제에서는 공백 기반 토크나이저, 정규 표현식 토크나이저, NLTK 토크나이저, SpaCy 토크나이저, BERT 토크나이저, Hugging Face 토크나이저 등을 비교합니다. 각 토크나이저는 속도, 정확도, 사용 편의성 등에서 차이가 있으므로, 이러한 비교는 실용적인 의미를 갖습니다.

텍스트의 길이를 늘리기 위해 이 내용을 반복하거나 추가적인 문장을 삽입할 수 있습니다. 예를 들어, 다음과 같은 문장을 추가할 수 있습니다.

"한국어는 형태소 분석이 중요한 언어 중 하나이며, 이를 위해 특화된 토크나이저가 필요합니다. 형태소 분석기는 어휘적, 문법적 정보를 활용하여 단어를 세분화하고 품사 태깅을 수행합니다."

또는 다양한 주제의 문장을 포함하여 토크나이저의 일반화 능력을 테스트할 수 있습니다.

"인공지능의 발전은 우리 삶의 많은 부분을 변화시키고 있으며, 의료, 금융, 교육 등 다양한 분야에서 활용되고 있습니다. 머신러닝 알고리즘은 데이터의 패턴을 학습하여 예측 모델을 생성합니다."

이러한 방식으로 텍스트를 구성하면, 토크나이저의 성능을 보다 정확하게 비교할 수 있습니다."""


nltk.donwload('punkt', quiet=True)

nlp = spacy.load("en_core_web_sm")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


new_tokenizer = Tokenizer(models.BPE())
new_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=30_000, min_frequency=2)
new_tokenizer.train_from_iterator([text], trainer=trainer)

tik_tokenizer = tiktoken.get_encoding("gpt2")


# 고정 규칙 기반 토크나이저
def whitespace_tokenizer(text):
    return text.split()

def regex_toknenizer(text):
    return re.findall(r'\b\w+\b', text)

def nltk_tokenizer(text):
    return nltk.word_tokenize(text)

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]

def bert_tokenizer(text):
    return bert_tokenizer.tokenize(text)

def gpt_tokenizer(text):
    return gpt_tokenizer.tokenize(text)

def hf_tokenizer(text):
    output = new_tokenizer.encoder(text)
    return output.tokens

def tiktokenizer(text):
    return tik_tokenizer.encode(text, allowed_special={"<|endoftext|>"})

tokenizers = {
    'whitespace': whitespace_tokenizer,
    'regex': regex_toknenizer,
    'nltk': nltk_tokenizer,
    'spacy': spacy_tokenizer,
    'bert': bert_tokenizer,
    'gpt': gpt_tokenizer,
    'hf': hf_tokenizer,
    'tiktoken': tiktokenizer,
}

if __name__ == "__main__":
    for name, tokenizer_func in tokenizers.items():
        start_time = time.time()
        tokens = tokenizer_func(kor_text)
        end_time = time.time()
        eplapsed_time = end_time - start_time
        print(f'{name}: {elapsed_time:.6f} seconds, {len(tokens)} tokens')