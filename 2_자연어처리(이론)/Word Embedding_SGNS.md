## Word Representatin 
> *word: 언어에서 의미를 갖는 최소 단위 'word'

### Word Encoding: 빈도기반, 통계적기반(단순 수치화)
	* TFIDF, Bow, Doc2Bow, Concurrance Matrix
	* 단점: 단어의 의미를 부여하지 못함
### Word Embedding: 학습기반 (학습을 통해 단어들을 수치 벡터로 변환)
	* 단어들의 관계, 단어의 의미 부여, 의미적인 유사성을 갖도록 수치화
	* 특정 목적을 달성하기 위해 그 때마다 학습하는 방식
	* vector는 사후적으로 결정되고, 특정 목적의 용도에 한정된다.
		ex)IMDB(영화관람후기, review:Pos, Neg)classification만을 하기 위한 embedding이라 범용적인 의		미가 아님  -> sensitive analysis

#### Word2Vec
	* 방대한 양의 코퍼스를 학습하여 어떤 관계를 갖도록 벡터화하는 기술
	* 목적에 상관없이 범용적(학습을 통해 사전제작하여 다른 학습에 적용)으로 사용할 수 있도록 벡터화한다.
	* 주변단어(context, 문맥)들을 참조하여 단어를 수치화 -> distributed representation
	* CBOW, Skip-Gram

##### Skip-Gram
	* 단점: 1. 동음이의어(bank)
				bank(money, invest)/ bank(river, water)의 평균값이 출력된다.
		   2. 출력층:softmax(one-hot) 계산량이 많다. => SGNS
		   3. OOV문제점

#### SGNS(Skip-Gram Negative Sampling)
	* softmax문제를 보완한 것, 계산량 감소를 위해 sigmoid를 사용한다.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk import pos_tag
from nltk.stem import PorterStemmer
import collections
from tensorflow.keras.layers import Input, Dense, Dropout, Dot
from tensorflow.keras.models import Model

# 전처리
def preprocessing(text):
    text2 = "".join([" " if ch in string.punctuation else ch for ch in text])
    tokens = nltk.word_tokenize(text2)
    tokens = [word.lower() for word in tokens]
    
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    
    tokens = [word for word in tokens if len(word)>=3]
    
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    tagged_corpus = pos_tag(tokens)    
    
    Noun_tags = ['NN','NNP','NNPS','NNS']
    Verb_tags = ['VB','VBD','VBG','VBN','VBP','VBZ']

    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token,tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token,'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token,'v')
        else:
            return lemmatizer.lemmatize(token,'n')
    
    pre_proc_text =  " ".join([prat_lemmatize(token,tag) for token,tag in tagged_corpus])             

    return pre_proc_text

# 소설 alice in wonderland를 읽어온다.
lines = []
fin = open("./dataset/alice_in_wonderland.txt", "r")
for line in fin:
    if len(line) == 0:  #lines = [line for line in fin if len(line)>]
        continue
    lines.append(preprocessing(line))
fin.close()

# 단어들이 사용된 횟수를 카운트 한다.
counter = collections.Counter()

for line in lines:
    for word in nltk.word_tokenize(line):
        counter[word.lower()] += 1

# 사전을 구축한다.
# 가장 많이 사용된 단어를 1번으로 시작해서 번호를 부여한다.
word2idx = {w:(i+1) for i,(w,_) in enumerate(counter.most_common())}
idx2word = {v:k for k,v in word2idx.items()}

# Trigram으로 학습 데이터를 생성한다.
inps = []     # 입력 데이터
tars = []

     # 출력 데이터
for line in lines:
    # 사전에 부여된 번호로 단어들을 표시한다.
    embedding = [word2idx[w.lower()] for w in nltk.word_tokenize(line)]
    
    # Trigram으로 주변 단어들을 묶는다.
    triples = list(nltk.trigrams(embedding))
    
    # 왼쪽 단어, 중간 단어, 오른쪽 단어로 분리한다.
    w_lefts = [x[0] for x in triples]
    w_centers = [x[1] for x in triples]
    w_rights = [x[2] for x in triples]
    f_target = [np.random.randint(1, 1786)]
    
    inps.extend(w_centers)
    tars.extend(w_lefts)
    inps.extend(w_centers)
    tars.extend(w_rights)
    inps.extend(w_centers)


# 학습 데이터를 one-hot 형태로 바꾸고, 학습용과 시험용으로 분리한다.
vocab_size = len(word2idx) + 1  # 사전의 크기   #1787
ohe = OneHotEncoder(categories = [range(vocab_size)])

#input word
word = ohe.fit_transform(np.array(inps).reshape(-1, 1)).todense()

#target word
target = ohe.fit_transform(np.array(tars).reshape(-1, 1)).todense()
f_rand = np.random.randint(1, 1786, size=(int(len(tars)/2),1))
f_target = ohe.fit_transform(f_rand.reshape(-1, 1)).todense()
target = np.concatenate((target, f_target),axis=0)


#label
label = np.zeros(target.shape[0])
label[ : int(target.shape[0]*2/3)] = 1    
label[int(target.shape[0]*2/3) : ] = 0 
        

Wtrain, Wtest, Ttrain, Ttest,Ltrain, Ltest = train_test_split(word, target, label, test_size=0.2)
print(Wtrain.shape, Wtest.shape, Ttrain.shape, Ttest.shape)

# 딥러닝 모델을 생성한다.
BATCH_SIZE = 128
NUM_EPOCHS = 20

#input
input_layer = Input(shape = (word.shape[1],), name="input")
input_emb = Dense(300, activation='relu',name = "input_emb")(input_layer)
input_dropout = Dropout(0.5, name="input_dropout")(input_emb)


#target
target_layer = Input(shape = (target.shape[1],), name="target")
target_emb = Dense(300, activation='relu',name = "target_emb")(target_layer)
target_dropout = Dropout(0.5, name="target_dropout")(target_emb)


#label
dot= Dot(axes=1)([input_dropout,target_dropout])
dotdense = Dense(1, activation='sigmoid')(dot)


model = Model([input_layer,target_layer], dotdense)
model.compile(optimizer = "rmsprop", loss="binary_crossentropy")

# 학습
hist = model.fit([Wtrain, Ttrain],Ltrain,
                 batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS,
                 shuffle=True,
                 validation_data =( [Wtest, Ttest],Ltest))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# Extracting Encoder section of the Model for prediction of latent variables
encoder = Model(input_layer, input_dropout)

# Predicting latent variables with extracted Encoder model
reduced_X = encoder.predict(Wtest)

##############################################################################

# 시험 데이터의 단어들에 대한 2차원 latent feature인 reduced_X를
# 데이터 프레임으로 정리한다.
final_pdframe = pd.DataFrame(reduced_X)
final_pdframe.columns = ["xaxis","yaxis"]
final_pdframe["word_indx"] = xsts
final_pdframe["word"] = final_pdframe["word_indx"].map(idx2word)

# 데이터 프레임에서 100개를 샘플링한다.
rows = final_pdframe.sample(n = 100)
labels = list(rows["word"])
xvals = list(rows["xaxis"])
yvals = list(rows["yaxis"])

# 샘플링된 100개 단어를 2차원 공간상에 배치한다.
# 거리가 가까운 단어들은 서로 관련이 높은 것들이다.
plt.figure(figsize=(15, 15))  

for i, label in enumerate(labels):
    x = xvals[i]
    y = yvals[i]
    plt.scatter(x, y)
    plt.annotate(label,xy=(x, y), xytext=(5, 2), textcoords='offset points',
                 ha='right', va='bottom', fontsize=15)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()

```