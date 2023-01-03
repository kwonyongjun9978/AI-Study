import tensorflow as tf  # 텐서플로를 입포트합니다. 하지만 너무 길어서 as로 뒤는줄여준다.
print(tf.__version__)
import numpy as np

# 1.데이터
x=np.array([1,2,3])
y=np.array([1,2,3])

# 2.모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()
model.add(Dense(1, input_dim=1)) #1 = y(output dim), input_dim=1 = x /dim = dimension(차원)


# 3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y, epochs=100)  #epochs(훈련을 시키는 횟수), 훈련을 많이 할수록 오차가 줄어듬, 너무 많이하면 과부하 발생

# 4.평가,예측
result=model.predict([4])
print('결과 : ', result)




 