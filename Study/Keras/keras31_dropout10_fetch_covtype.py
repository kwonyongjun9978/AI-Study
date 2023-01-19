import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets['target']
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840,283301,35754,2747,9493,17367,20510],dtype=int64))

# 원핫인코딩
#방법3(scikit-onehotencoder)
# print(y.shape) #(581012,)
# print(type(y)) #<class 'numpy.ndarray'>
y = y.reshape(581012, 1)
# print(y.shape) #(581012, 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
# print(y[:15])
# print(type(y)) #<class 'scipy.sparse._csr.csr_matrix'>
# print(y.shape) #(581012, 7)
y=y.toarray()
# print(y[:15])
# print(type(y)) #<class 'numpy.ndarray'>
# print(y.shape) #(581012, 7)

'''
방법3
OneHotEncoder : 명목변수든 순위변수든 모두 원핫인코딩을 해준다.
=> 해결방법: shape 맞추기
1) 스칼라: 원본 데이터를 y, y.shape, type(y)를 print 해보면
(581012,) 스칼라 형태의 numpy.ndarray 임을 알 수 있다.
2) 벡터: 원핫엔코더하려면 벡터 형태로 reshape 해줘야 한다.
y = y.reshape(581012,1) 해서 (581012, 1) 벡터 형태를 만든다.
# (-1,1) 하면 (전체, 1)과 같다.
3) scikit-learn에서 OneHotEncoder 가져오기
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
4) 원핫엔코딩: y = ohe.fit_transform(y)로 원핫엔코딩한다.
y = ohe.fit_transform(y) 하면 (581012, 7) 벡터 형태의 scipy.sparse._csr.csr_matrix가 나온다.
5) 데이터형태 바꾸기 : scipy CSR matrix 를 Numpy ndarray로 바꾼다.
y = y.toarray() 하면 데이터 종류만 numpy ndarray로 바뀐다.
'''

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,     
    #random_state=333
    stratify=y 
)

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2.모델구성(함수형)
input1=Input(shape=(54,))
dense1=Dense(256, activation='relu')(input1)
drop1=Dropout(0.5)(dense1)
dense2=Dense(128, activation='relu')(drop1)
drop2=Dropout(0.3)(dense2)
dense3=Dense(64, activation='relu')(drop2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
output1=Dense(7, activation='softmax')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
import time 
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, 
                              restore_best_weights=True, 
                              verbose=2)

import datetime
date = datetime.datetime.now() 
print(date) 
print(type(date)) 
date=date.strftime("%m%d_%H%M") 
print(date) 
print(type(date)) 

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=2,
                                  save_best_only=True,
                                  filepath=filepath+'k31_10_'+date+'_'+filename)

start=time.time() 
model.fit(x_train, y_train, epochs=500, batch_size=3000,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=2)
end=time.time()

#4 평가, 예측
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

from sklearn.metrics import accuracy_score

y_predict=model.predict(x_test)

y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict[:20])
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test[:20])
acc=accuracy_score(y_test, y_predict)
print("acc(정확도) : ", acc)

print("걸린시간 : ", end-start)

'''
loss :  0.24077723920345306
accuracy :  0.9054241180419922
acc(정확도) :  0.9054241284648417
'''
