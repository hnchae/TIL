## 챗봇만들기

### seq2seq-Attention model

![seq2seq_Attention model](C:%5CUsers%5Cstudent%5CDownloads%5Cseq2seq_Attention%20model.jpeg)



```python
from tensorflow.keras.layers import Input, LSTM, concatenate,Dense, Dot, Multiply
from tensorflow.keras.layers import Embedding, TimeDistributed, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import pickle
import numpy as np

#출처: https://neurowhai.tistory.com/178 [NeuroWhAI의 잡블로그]

# 단어 목록 dict를 읽어온다.
with open('./dataset/6-1.vocabulary.pickle', 'rb') as f:
    word2idx,  idx2word = pickle.load(f)
    
# 학습 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 읽어온다.
with open('./dataset/6-1.train_data.pickle', 'rb') as f:
    trainXE, trainXD, trainYD = pickle.load(f)
	
# 평가 데이터 : 인코딩, 디코딩 입력, 디코딩 출력을 만든다.
with open('./dataset/6-1.eval_data.pickle', 'rb') as f:
    testXE, testXD, testYD = pickle.load(f)

VOCAB_SIZE = len(idx2word) #+1을 안하는 이유는 OOV(unknown)의 1이 추가가 이미 되어있기 때문에
EMB_SIZE = 128
LSTM_HIDDEN = 128
E_SIZE = 128
MODEL_PATH = './dataset/6-2.Seq2Seq.h5'
LOAD_MODEL = True #추가학습 / False=처음부터학습

# 워드 임베딩 레이어. Encoder와 decoder에서 공동으로 사용한다.
K.clear_session()
wordEmbedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMB_SIZE)

# Encoder
# -------
# many-to-one으로 구성한다. 중간 출력은 필요 없고 decoder로 전달할 h와 c만
# 필요하다. h와 c를 얻기 위해 return_state = True를 설정한다.
encoderX = Input(batch_shape=(None, trainXE.shape[1]))
encEMB = wordEmbedding(encoderX)
encLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
encLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state = True)
ey1, eh1, ec1 = encLSTM1(encEMB)    # LSTM 1층 
ey2, eh2, ec2 = encLSTM2(ey1)         # LSTM 2층


# Decoder
# -------
# many-to-many로 구성한다. target을 학습하기 위해서는 중간 출력이 필요하다.
# 그리고 초기 h와 c는 encoder에서 출력한 값을 사용한다 (initial_state)
# 최종 출력은 vocabulary의 인덱스인 one-hot 인코더이다.
decoderX = Input(batch_shape=(None, trainXD.shape[1]))
decEMB = wordEmbedding(decoderX)
decLSTM1 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
decLSTM2 = LSTM(LSTM_HIDDEN, return_sequences=True, return_state=True)
dy1, _, _ = decLSTM1(decEMB, initial_state = [eh1, ec1])
dy2, _, _ = decLSTM2(dy1, initial_state = [eh2, ec2])

#1번(Dot), 2번(softmax취하기) 
att_dot = Dot(axes=2)([dy2,ey2]) # Dot(axis=[2,2])([dy2,ey2])
att_act = Activation(activation='softmax')(att_dot) 

#내가 만든 3번
# att_mul1 = K.expand_dims(att_act)
# att_mul2 = att_mul1 * ey2
# att_mul3 = K.sum(att_mul2*ey2, axis=1)

#3번(attention scor), 4번(sum) 합한것 => Dot을 해주면 됐다 / 5번(concat)
att_mul_sum =  Dot(axes=(2, 1))([att_act, dy2])
att_con = concatenate([att_mul_sum, dy2],axis=2)

decOutput = TimeDistributed(Dense(VOCAB_SIZE, activation='softmax'))
outputY = decOutput(att_con)

# Model
# -----
# target이 one-hot encoding되어 있으면 categorical_crossentropy
# target이 integer로 되어 있으면 sparse_categorical_crossentropy를 쓴다.
# sparse_categorical_entropy는 integer인 target을 one-hot으로 바꾼 후에
# categorical_entropy를 수행한다.
model = Model([encoderX, decoderX], outputY)
model.compile(optimizer=optimizers.Adam(lr=0.001), 
              loss='sparse_categorical_crossentropy')

if LOAD_MODEL:
    model.load_weights(MODEL_PATH)
    
# 학습 (teacher forcing)
# ----------------------
# loss = sparse_categorical_crossentropy이기 때문에 target을 one-hot으로 변환할
# 필요 없이 integer인 trainYD를 그대로 넣어 준다. trainYD를 one-hot으로 변환해서
# categorical_crossentropy로 처리하면 out-of-memory 문제가 발생할 수 있다.
hist = model.fit([trainXE, trainXD], trainYD, batch_size = 300, 
                 epochs=100, shuffle=True,
                 validation_data = ([testXE, testXD], testYD))

# Loss history를 그린다
plt.plot(hist.history['loss'], label='Train loss')
plt.plot(hist.history['val_loss'], label = 'Test loss')
plt.legend()
plt.title("Loss history")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

# 학습 결과를 저장한다
model.save_weights(MODEL_PATH)
```



