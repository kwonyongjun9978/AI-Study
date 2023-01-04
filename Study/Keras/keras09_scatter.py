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
model.compile(loss='mae', optimizer='adam') #mae = mean(평균),absolute(절대값),error
model.fit(x_train, y_train, epochs=200, batch_size=1)

#4.평가,예측
loss=model.evaluate(x_test, y_test) #test loss가 train loss보다 안좋다.
print('loss : ', loss)

y_predict=model.predict(x)

import matplotlib.pyplot as plt
plt.scatter(x,y)  #산점도 (Scatter plot)는 두 변수의 상관 관계를 직교 좌표계의 평면에 점으로 표현하는 그래프
plt.plot(x, y_predict, color='red')
plt.show()



