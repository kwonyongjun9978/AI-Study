import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense      

#1.데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

#2.모델구성(hidden layer 구성 변경(노드,layer 의 개수)->하이퍼 파라미터 튜닝)
model=Sequential()  #Sequential api를 모델이라는 이름으로 정의
model.add(Dense(3, input_dim=1))  
model.add(Dense(5))  #input layer는 표시하지 않아도 된다.  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(4))  #hidden layer
model.add(Dense(2))  #hidden layer
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=299)

# 4. 평가, 예측
results=model.predict([6])
print('6의 예측값 : ', results)
