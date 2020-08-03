### Cross Entropy 와 Loss Function

* 문제의 목적에 맞게 BCE 와 CCE 를 알맞게 선택해야 한다.
* 출력에 '1' 이 여러 개인 경우 (각각 sigmoid 출력)는 BCE 를 사용하고 , 출력에 '1' 이 한 개인 경우
  (one hot 형태- softmax 출력) 는 CCE 를 사용한다. 
* BCE 와 CCE 는 결과가 다르고 정확도 측정도 달라지므로 주의해서 선택해야 한다.



1. Binary_classification :              "sigmoid"  ==  "binary_cross_entropy"

2. multi_classification :                "softmax"  ==  "categorical_cross_entropy" (one-hot)

3. multi_labeled classification :  "sigmoid"  ==  "binary_cross_entropy"         (one-hot 구조가 아니다)

   => 각각에 대해 binary_classficiation을 취해준다.



> 1. ​												2.													3.	
>
>    y         y_hat 							y           y_hat 								y            y_hat 
>
>    0          0							     0 1 0       0 1 0      						 0 1 0        0 1 0  
>
>    0		  1 								0 0 1 	  0 1 0 							  0 0 1	    0 1 0
>
>    1		  1 								1 0 0	   1 0 0							   1 0 0		1 0 0
>
>    *acc = 2/3 								*acc = 2/3 									*acc = 7/9
>
> 

##### 실습파일 10-3 Predict Next Word.py

###### 과제2 : [ㅇㅇㅇ##]  일 경우 연속된 'o' 3 단어 뒤, 연속된 '#' 2개 단어 예측

```python
# Select value of N for N grams among which N-1 are used to predict
# last N word
N = 5
quads = list(nltk.ngrams(tokens,N))

newl_app = []
for ln in quads:
    newl = " ".join(ln)        
    newl_app.append(newl)

    
# Vectorizing the words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

x_trigm = []
y_trigm = []

for l in newl_app:
    x_str = " ".join(l.split()[0:N-2])
    y_str = " ".join(l.split()[N-2:])
    x_trigm.append(x_str)
    y_trigm.append(y_str)

x_trigm_check = vectorizer.fit_transform(x_trigm).todense()
y_trigm_check = vectorizer.fit_transform(y_trigm).todense()


# Model Building
BATCH_SIZE = 128
NUM_EPOCHS = 100

input_layer = Input(shape = (Xtrain.shape[1],),name="input")
first_layer = Dense(1000,activation='relu',name = "first")(input_layer)
first_dropout = Dropout(0.5,name="firstdout")(first_layer)

second_layer = Dense(800,activation='relu',name="second")(first_dropout)

third_layer = Dense(1000,activation='relu',name="third")(second_layer)
third_dropout = Dropout(0.5,name="thirdout")(third_layer)

fourth_layer = Dense(Ytrain.shape[1],activation='sigmoid',name = "fourth")(third_dropout)



history = Model(input_layer,fourth_layer)
history.compile(optimizer = "adam",loss="binary_crossentropy",metrics=["accuracy"])
```





 