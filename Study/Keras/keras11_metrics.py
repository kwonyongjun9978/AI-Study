from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x=np.array(range(1,21))
y=np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=123
)

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse', 'acc'])  
model.fit(x_train, y_train, epochs=200, batch_size=1)
# 머신러닝 회귀 모델의 성능 평가 지표                                  
# ['mae(평균 절대 오차)', 'mse(평균 제곱 오차)', 'acc(accuracy)(정확도)']
# 0에 가까울 수록 좋은 성능(mae,mse,rmse,loss)
# 1에 가까울 수록 좋은 성능(acc,R2)
# loss는 훈련에 영향을 미친다. → 가중치 갱신에 영향을 미침
# metric는 훈련에 영향을 미치지 않는다.
# loss: 손실함수. 훈련셋과 연관. 훈련에 사용. 
# metric: 평가지표. 검증셋과 연관. 훈련 과정을 모니터링하는데 사용.

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

#mae :  3.143047332763672
#mse :  14.79165267944336



