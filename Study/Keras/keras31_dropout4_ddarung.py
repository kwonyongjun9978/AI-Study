import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path='./_data/ddarung/' 

train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'submission.csv', index_col=0) 

#선형 방법을 이용하여 결측치
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x=train_csv.drop(['count'], axis=1)
y=train_csv['count']

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=444
)

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

test_csv=scaler.transform(test_csv)

#2.모델구성
inputs = Input(shape=(9, ))
hidden1 = Dense(256, activation='relu') (inputs)
drop1=Dropout(0.5)(hidden1)
hidden2 = Dense(128, activation='relu') (drop1)
drop2=Dropout(0.3)(hidden2)
hidden3 = Dense(64, activation='relu') (drop2)
hidden4 = Dense(32, activation='relu') (hidden3)
hidden5 = Dense(16, activation='relu') (hidden4)
hidden6 = Dense(8, activation='relu') (hidden5)
output = Dense(1, activation='relu') (hidden6)
model = Model(inputs=inputs, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=300, 
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
                                  verbose=1,
                                  save_best_only=True,
                                  filepath=filepath+'k31_4_'+date+'_'+filename)

hist=model.fit(x_train, y_train, epochs=10000, batch_size=8, 
               validation_split=0.25
               ,callbacks=[earlyStopping, ModelCheckpoint], verbose=2)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))              
print("RMSE : ", RMSE(y_test, y_predict)) 

#제출
y_submit=model.predict(test_csv)

submission['count']=y_submit

submission.to_csv(path+'submission_011801.csv')

'''
Epoch 00546: val_loss did not improve from 30.81214
Epoch 00546: early stopping
10/10 [==============================] - 0s 555us/step - loss: 26.5893
loss :  26.5893497467041
RMSE :  41.109200287913296
'''

