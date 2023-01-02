import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([[1,2,3,4,5,6,7,8,9,10], [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4], [9,8,7,6,5,4,3,2,1,0]])
y=np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)  #(3,10) 3행10열
print(y.shape)  #(10,)

x=x.T # T : 행과 열을 바꿈
print(x.shape)  #(10,3) 10행3열

#2.모델구성
model=Sequential() #순차적 모델을 만든다
model.add(Dense(5, input_dim=3)) # input_dim = 열의 개수
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
results=model.predict([[10, 1.4, 0]])
print('[10, 1.4, 0]의 예측값 : ', results)


'''
loss :  0.04214806482195854
[10, 1.4, 0]의 예측값 :  [[19.99323]]
'''
