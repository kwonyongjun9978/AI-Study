from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset=load_boston()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)

scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2.모델구성(함수형)
input1=Input(shape=(13,))
dense1=Dense(256, activation='relu')(input1)
dense2=Dense(128, activation='relu')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
output1=Dense(1, activation='relu')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary() #Total params: 47,361

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint #대문자=class, 소문자=함수 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=50, 
                              restore_best_weights=False, 
                              verbose=1 )  

import datetime
date = datetime.datetime.now() #현재 시간 반환
print(date) #2023-01-12 15:02:49.383746
print(type(date)) #<class 'datetime.datetime'>
date=date.strftime("%m%d_%H%M") #문자열 타입으로 변환
print(date) #0112_1502
print(type(date)) #<class 'str'>

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' #0037-0.0048.hdf8 
# {} 안에 있는 건 값 가져오라는 뜻임. 일반적인 문자와 다름.
# epoch의 점수 4자리, loss의 소수점 4자리까지 있는 모델 파일명

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  #filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5'
                                  filepath=filepath+'k30_'+date+'_'+filename)
                                
                                                                
hist = model.fit(x_train, y_train, epochs=5000, batch_size=2, 
                 validation_split=0.2, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=1)  

#4.평가,예측
# print("=========================1. 기본 출력===========================")
mse=model.evaluate(x_test, y_test) 
print('mse : ', mse)

from sklearn.metrics import mean_squared_error, r2_score
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)










                  



