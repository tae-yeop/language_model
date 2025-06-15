Tokenization

한국어의 경우 형태소 분석기를 통해 접사를 떼어놓고 BPE를 쓰거나 그냥 BPE를 쓰기도 한다

영어는 Space를 기준으로 word가 분할되지만 한국어는 White space만으로 분리가 힘듬. 그래서 형태소를 기반으로 tokenizing을 해야함. 이후 구글에서 제시한 WordPiece라는 방식은 한국어 등 다른 언어에도 적용가능함.

라이브러리는 konlpy나 속도가 빠른 mecab 사용

```
# https://github.com/SOMJANG/Mecab-ko-for-Google-Colab

! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh

from konlpy.tag import Mecab
mecab = Mecab()

mecab.morphs('동해물과 백두산이 마르고 닳도록')

>>> ['동해', '물', '과', '백두산', '이', '마르', '고', '닳', '도록']
```


- 한국어는 교착어 ⇒ 어근에 어미/ 접사가 붙어서 역할과 의미를 구성
- 이로 인해 같은 단어에서 여러가지 단어가 파생됨
- 따라서 어미/접사를 분리해준다 (안그러면 수십개로 늘어난다)
- 이를 위해 형태소 분석기를 이용해서 떼어낸다
- 형태소 기반 모델에게 데이터를 넣기 전엔 형태소 분석이 필요.

한국어 문장에 대한 형태소 분석의 방법은 크게 두 가지

첫 번째는 형태소 분석 패키지를 사용하는 것입니다. 파이썬에서는 KoNLPy에 포함되어 있는 Komoran, Okt, MeCab 등으로 형태소 분석을 해볼 수 있습니다.

둘째로 형태소 분석 API를 사용하는 방법이 있습니다. 여기에선 ETRI의 형태소 분석 API를 사용할 예정입니다. 형태소 분석의 결과가 비교적 정확하고, 태그셋이 동일 기관의 BERT과 일치하는 점을 고려하여 선택하였습니다.