import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array(range(10))   
# print(x.shape)    // (10,)
y=np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],[9,8,7,6,5,4,3,2,1,0]])
y=y.T #(10,3)

#2.모델구성
model=Sequential() 
model.add(Dense(11, input_dim=1)) 
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))


#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=300, batch_size=1)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
results=model.predict([9])
print('[9]의 예측값 : ', results)


'''
loss :  0.06930460780858994
[9]의 예측값 :  [[10.029289    1.6872673   0.05623817]]
'''