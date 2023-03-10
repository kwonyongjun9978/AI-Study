from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset=fetch_california_housing()
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

# 2.모델구성(함수형)
input1=Input(shape=(8,))
dense1=Dense(50, activation='relu')(input1)
dense2=Dense(100, activation='relu')(dense1)
dense3=Dense(150, activation='relu')(dense2)
dense4=Dense(200, activation='relu')(dense3)
dense5=Dense(250, activation='relu')(dense4)
dense6=Dense(300, activation='relu')(dense5)
drop1=Dropout(0.3)(dense6)
dense7=Dense(350, activation='relu')(drop1)
drop2=Dropout(0.4)(dense7)
dense8=Dense(400, activation='relu')(drop2)
drop3=Dropout(0.5)(dense8)
dense9=Dense(350, activation='relu')(drop3)
drop4=Dropout(0.4)(dense9)
dense10=Dense(300, activation='relu')(drop4)
drop5=Dropout(0.3)(dense10)
dense11=Dense(250, activation='relu')(drop5)
dense12=Dense(200, activation='relu')(dense11)
dense13=Dense(150, activation='relu')(dense12)
dense14=Dense(100, activation='relu')(dense13)
dense15=Dense(50, activation='relu')(dense14)
output1=Dense(1, activation='relu')(dense15)
model=Model(inputs=input1, outputs=output1)
model.summary() #Total params: 843,651

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=500, 
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
                                  filepath=filepath+'k31_2_'+date+'_'+filename)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=200, 
                 validation_split=0.2, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=2)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
Epoch 00563: val_loss did not improve from 0.28264
Epoch 00563: early stopping
129/129 [==============================] - 0s 2ms/step - loss: 0.2827
loss :  0.2827185094356537
RMSE :  0.5317127633608937
R2 :  0.7780995739992294
'''
                  



