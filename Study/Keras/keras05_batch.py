import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense      

#1.데이터
x=np.array([1,2,3,4,5,6]) 
y=np.array([1,2,3,5,4,6])

#2.모델구성
model=Sequential()  
model.add(Dense(3, input_dim=1))  
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=2) #하이퍼 파라미터 튜닝 - 2. batch_size 조절
                                       #6개를 2개단위로 나누어서 훈련(batch),3/3
                                       #batch size default(기본값)=32
                                       
                                    
# 4. 평가, 예측
results=model.predict([6])
print('6의 예측값 : ', results)

#블럭주석처리 : 큰따움표or작은따움표3개
"""
batch size=1 : 6개를 1개단위로 나누어서 훈련(batch),6/6 
batch size=2 : 6개를 2개단위로 나누어서 훈련(batch),3/3 
batch size=3 : 6개를 3개단위로 나누어서 훈련(batch),2/2 
batch size=4 : 6개를 4개단위로 나누어서 훈련(batch),2/2 
batch size=5 : 6개를 5개단위로 나누어서 훈련(batch),2/2 
batch size=6 : 6개를 6개단위로 나누어서 훈련(batch),1/1 
"""
