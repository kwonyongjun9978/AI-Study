import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense      

#1.데이터
x=np.array([1,2,3,4,5,6]) 
y=np.array([1,2,3,5,4,6])

#2.모델구성(hidden layer 구성 변경(노드,layer 의 개수)->하이퍼 파라미터 튜닝(1,2))
model=Sequential()  #Sequential api를 모델이라는 이름으로 정의
model.add(Dense(3, input_dim=1))  
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=10, batch_size=2) #6개를 2개단위로 나누어서 훈련(batch),3/3-->훈련 시간이 길어짐, 자원 소비가 크다
                                       #batch size default=32(기본값)
                                       #하이퍼 파라미터 튜닝(3)
                                    
# 4. 평가, 예측
results=model.predict([6])
print('6의 예측값 : ', results)

#블럭주석처리:큰따움표or작은따움표3개
"""
batch
6개를 1개단위로 나누어서 훈련(batch),6/6 
6개를 2개단위로 나누어서 훈련(batch),3/3 
6개를 3개단위로 나누어서 훈련(batch),2/2 
6개를 4개단위로 나누어서 훈련(batch),2/2 
6개를 5개단위로 나누어서 훈련(batch),2/2 
6개를 6개단위로 나누어서 훈련(batch),1/1 
"""
