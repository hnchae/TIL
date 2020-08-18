## BERT

(= Bidirectional Encoder Representations Transformer)



BERT 논문정리 링크

[한글정리]: https://mino-park7.github.io/nlp/2018/12/12/bert-%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0G




## Abstract

* 모든 layer에서 양방향(left and right) context에서 공동으로 조절하여 unlabeled text에서 pre-train deep bidirectional representations으로 설계

* BERT는 기본적으로, 대용량(wiki나 book data) **unlabeled data**로 모델을 미리 학습 시킨 후, 특정 *task*를 가지고 있는 *labeled data*로 **transfer learning**을 하는 모델입니다.
*  Pre-trained BERT은 output layer를 하나를 추가해서 fine-tune할 수 있습니다. 실제 task-specific 구조 수정 없이 BERT는 여러 분야에서 SOTA를 달성했습니다.

## 1. Introduction

## 2. Related Work
### 2.1 Unsupervised Feature-based Approaches
### 2.2 Unsupervised Fine-tuning Approaches
### 2.3 Transfer Learning from Supervised Data

## 3. BERT

### 3.1 pretraining BERT
unsupervised


![image-20200818134646033](C:%5CUsers%5Cstudent%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5Cimage-20200818134646033.png)

> 	* [cls] 토큰 - 문장의 시작을 알린다.
>	
> 	* [sep] 토큰 - 문장의 나뉨을 알린다.
>	
> 	* Token Embeddings   - 단어들의 토큰을 이용한 Embedding
>	
> 	* Segment Embeddings - A라는 문장과 B라는 문장을 구별하기 위한 Embedding
>	
> 	* Position Embeddings - 단어들의 위치정보를 이용한 Embedding

* playing => play + ing : word piece (sentencepiece)



#### Task1: Masked LM= 중간의 단어를 [Mask]를 대치한 후 원래 단어가 나오도록 학습 
bidirectional이 단방향보다 낫다.(단방향 두개를 (오->왼, 왼-> 오)concat 해놓은것)
LM : p(xt| x1, x2,,,xt-1)
		p(xt|x1,x2,,,xt-1, xt+1,xt+2)
	** 단점:  [mask]는 독립가정이다.

##### Pre-training Tasks - Language model을 학습시키기 위해 필요

* MLM (Masked Language Model)
	* 문장 내 랜덤 한 단어를 masking, 예측하도록 하는 방식
        * input에서 랜덤 하게 몇 개의 token을 mask
            * Transformer 구조에 넣어 주변 단어의 context만을 보고 mask 된 단어 예측
            * Input 전체와 mask된 token을 한 번에 Transformer encoder에 넣고 원래 token값을 예측
    

* 단어중 일부(15%)를 [mask] token으로 변경

  - 15% 중 80% token을 [mask]로 변경
  - 15% 중 10% token을 random word로 변경
  - 15% 중 10% token을 원래의 단어 그대로(실제 관측된 단어에 대한 표상을 bias 해주기 위해)



#### Task2: Next Sentence Prediction(NSP)

##### 목적: 두 문장의 유사도(맨하탄)를 classification을 하기위한것

- 두 문장을 pre-training시 같이 넣어 두 문장이 이어지는 문장인지 아닌지를 맞추는 작업
- pre-training시 50:50 비율로 실제 이어지는 두 문장과 랜덤 하게 추출된 두 문장을 넣고 BERT가 맞추도록 하는 작업


### 3.2 Fine-tuning BERT




## 4. EXperiments

### 4.1 GLUE
### 4.2 SQuAD v1.1
### 4.3 SQuAD v2.0
### 4.4 SWAG

## 5. Ablation Studies
### 5.1 Effect of Pre-training Tasks
### 5.2 Effect of Model Size
### 5.3 Feature-based Approach with BERT

## 6. Conclusion



## A) Additional Details for BERT





## BERT 실습

* `keras_bert`검색 후,  https://github.com/CyberZHG/keras-bert에 나온 'code download' 한다.

* https://github.com/google-research/bert의 원하는 file download 한다.















































































