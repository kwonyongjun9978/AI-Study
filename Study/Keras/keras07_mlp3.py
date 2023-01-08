import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([range(10), range(21,31), range(201,211)]) # 0 ~ (x-1), 0~9
#print(x.shape) //(3,10)
y=np.array([[1,2,3,4,5,6,7,8,9,10], [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
x=x.T #(10,3)
y=y.T #(10,2)

#2.모델구성
model=Sequential() 
model.add(Dense(10, input_dim=3)) 
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))


#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=250, batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
results=model.predict([[9, 30, 210]])
print('[10, 1.4, 0]의 예측값 : ', results)

'''
loss :  0.09067851305007935
[10, 1.4, 0]의 예측값 :  [[9.973186  1.5327263]]
'''