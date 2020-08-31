# Wordpiece, Byte pair Encoding, Sentence piece



* 자연어 처리 시 모든 단어들을 벡터화 하는데 한계가 있다.( 이름, 숫자, 희귀단어, 단어장에 없는 신종어, 변형어, 약어 등등)  -> Rare word problem에서  Out-of-vocabulary(OOV) 문제가 발생 -> 보완 -> Sub-word 전략사용

* 




## Wordpiece









## BPE(Byte Pair Encoding)

* 서브워드(sub-word) 단위 분절 방법 => NMT, BERT 등 최근 자연어처리 알고리즘에서 전처리로 이용
* 단어는 의미를 가진 더 작은 서브워드들의 조합으로 이루어진다. 빈도수에 따라 문자를 병합하여 subword를 구성 (예를 들면 "conference" -> 'con', 'f', 'er', 'ence' 로 분절 할 수 있다.)

* 한국어, 일본어, 영어 등 언어 형식에 구애받지 않고, 별도의 문법 입력 없이 조사 구분이 가능

* 장점 - 어휘에서 자유롭게 학습이 가능하고, Rare word를 위한 back-off model이 필요 없음

  ​		 - 성능 향상 효과도 있음

  ​		 - 서브워드 단위로 쪼개면 차원과 sparsiry도 줄일 수 있음

  ​		 - 처음 본 단어(unknown word)와 OOV 처리에 효과적

* 단점 - 조사가 어색할 수 도 있음

### 1) 모델만들기

- 가지고 있는 text로 빈도수 기반 BPE 모델을 만들 수 있다.

- 파이썬을 이용하여 sentencepiece 를 호출하고, [spm.SentencePieceTrainer.Train]('--input=test/test.txt) 함수를 실행한다.

     -- input : 학습시킬 text의 위치

    -- model_prefix : 만들어질 모델 이름, 학습 후 <model_name>.model, <model_name>.vocab 두 파일 생성

    -- vocab_size : Subword 갯수 설정, 보통 3,200개

    -- character_coverage : 몇%의 데이터를 커버할것인가, default=0.9995, 데이터셋이 적은 경우 1.0으로 설정

   -- model_type : 어느 방식으로 학습할것인가, (unigram(default), bpe, char, word)

![image-20200817113553624](C:%5CUsers%5Cstudent%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20200817113553624.png)



### 2) BPE 분절하기

* 1) 과정 에서 가지고 있는 text로 BPE 모델을 만들었다.

* 이 모델(model_prefix + '.model)을 이용하여 분절을 수행한다.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.Train(params)
sp = spm.SentencePieceProcessor()
sp.Load(model_prefix + '.model')

sp.GetPieceSize() # 9000
```

























