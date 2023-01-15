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
model.fit(x,y,epochs=100, batch_size=2) 
#가중치(w) 생성(y=wx+b)
                                       
# 4. 평가, 예측
# loss 값 (실제값과 예측값의 오차) 반환 → 모델의 평가 기준 = (predict보다)loss (0과 가까울수록 좋다)
loss=model.evaluate(x,y) 
print('loss : ', loss)
results=model.predict([6])
print('6의 예측값 : ', results)

'''
evaluate의 batch_size의 default(기본값)도 32이다.
그래서 마지막에 1/1 나온다.
'''


