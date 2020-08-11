###### text를 수치 vector로 표현하는 방법에는 2가지가 있다.

## 1.통계적기반

### 빈도기반, 카운트 기반

#### TFIDF. Bow(Bag of word)
##### Bow
		I love you very much, you love me too
			1. vocabulary 생성
				I: 0, love: 1, you: 2 ...
				1) [0, 1, 2, ...2, 1, ...] <- vector
				2) [(0,1), (1,2), ..., ] (voca, 빈도) / 압축 <- vector 
##### python에서 gensim을 설치 후, doc2bow를 사용할 수 있다! 

## 2. Embedding 기반
#### 장점: 단어의 의미를 살릴 수 있다.





