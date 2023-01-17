from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input,Dropout
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
    random_state=333
)

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2.모델구성(순차형)
model=Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))
model.summary() 

# 2.모델구성(함수형)
# input1=Input(shape=(13,))
# dense1=Dense(256, activation='relu')(input1)
# drop1=Dropout(0.5)(dense1)
# dense2=Dense(128, activation='relu')(drop1)
# drop2=Dropout(0.3)(dense2)
# dense3=Dense(64, activation='relu')(drop2)
# drop3=Dropout(0.2)(dense3)
# dense4=Dense(32, activation='relu')(drop3)
# dense5=Dense(16, activation='relu')(dense4)
# output1=Dense(1, activation='relu')(dense5)
# model=Model(inputs=input1, outputs=output1)
# model.summary() #Total params: 47,361

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=200, 
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
                                  filepath=filepath+'k31_1_'+date+'_'+filename)
                                                                                           
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, 
                 validation_split=0.2, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=2)  

#4.평가,예측
mse=model.evaluate(x_test, y_test) 
print('mse : ', mse)

from sklearn.metrics import mean_squared_error, r2_score
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
Epoch 00392: val_loss did not improve from 5.80611
Epoch 00392: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 19.3484 - mae: 2.5393
mse :  [19.34838104248047, 2.5393197536468506]
R2 :  0.8027265452002512
'''








                  



