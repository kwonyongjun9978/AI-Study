from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model,load_model
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
    random_state=333
)

scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2.모델구성(함수형)

#3.컴파일, 훈련

#모델+가중치save
path='./_save/'
# model.save(path + 'keras29_03_save_model.h5')
#R2 :  0.8579221121943994

#모델+가중치 불러오기
model=load_model(path+'keras29_03_save_model.h5')
#R2 :  0.8579221121943994

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)








                  



