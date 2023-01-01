import tensorflow as tf  
print(tf.__version__)
import numpy as np

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

# [13]예측해보기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(1, input_dim=1)) #1 = y imput_dim=1 = x


# 3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=2000)  #epochs(훈련을 시키는 횟수)

# 4.평가,예측
result=model.predict([13])
print('결과 : ', result)




 
