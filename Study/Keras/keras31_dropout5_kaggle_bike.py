import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path='./_data/bike/' 
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

train_csv = train_csv.dropna()

x=train_csv.drop(['casual','registered','count'], axis=1)
y=train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
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
print(x_train.shape)

# 2.모델구성(함수형)
input1=Input(shape=(8,))
dense1=Dense(32, activation='relu')(input1)
dense2=Dense(256, activation='relu')(dense1)
drop1=Dropout(0.4)(dense2)
dense3=Dense(512, activation='relu')(drop1)
drop2=Dropout(0.5)(dense3)
dense4=Dense(128, activation='relu')(drop2)
drop3=Dropout(0.2)(dense4)
dense5=Dense(16, activation='relu')(drop3)
output1=Dense(1, activation='relu')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary() 

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=300, 
                              restore_best_weights=True, 
                              verbose=2)  

hist = model.fit(x_train, y_train, epochs=10000, batch_size=64,
                 validation_split=0.2, callbacks=[earlyStopping], 
                 verbose=2)

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
loss :  20308.974609375
RMSE :  142.50955986002919
'''



