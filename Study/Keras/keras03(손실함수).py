import numpy as np
import tensorflow as tf

# 1. (정제된)데이터 
# (데이터 전처리 = 데이터를 분석 및 처리에 적합한 형태로 만드는 과정, 정제되지 않은 데이터를 정제시킴)
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

# 2. 모델구성(y=wx+b)
from tensorflow.keras.models import Sequential # Sequential : 딥러닝 모델에서 순차적인 모델을 만들수 있다
from tensorflow.keras.layers import Dense      # Dense : y=wx+b를 구성

model=Sequential()
model.add(Dense(1, input_dim=1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # 손실함수 mae
                                            # cf)손실함수란? https://heytech.tistory.com/361
model.fit(x,y,epochs=200)

# 4. 평가, 예측
results=model.predict([6])
print('6의 예측값 : ', results)