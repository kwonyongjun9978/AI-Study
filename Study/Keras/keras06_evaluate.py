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
model.fit(x,y,epochs=10, batch_size=7) 
#가중치(w) 생성(y=wx+b)
                                       
# 4. 평가, 예측
loss=model.evaluate(x,y) #loss값이 반환된다.
print('loss : ', loss)
results=model.predict([6])
print('6의 예측값 : ', results)

#판단의 기준은 loss(0과 가까울수록 좋다)(predict보다)


