'''
실습
R2 0.55~0.6 이상
'''
from  sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
dataset=fetch_california_housing()
x=dataset.data
y=dataset.target

'''
print(x)
print(x.shape) #(20640, 8)
print(y)
print(y.shape) #(20640, )

print(dataset.feature_names)
print(dataset.DESCR)
'''

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.75,
    shuffle=True,
    random_state=44
)

#2.모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(512))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=500, batch_size=50)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
loss :  [0.6031068563461304, 0.5825958847999573]
RMSE :  0.776599474530112
R2 :  0.5525065886459053
'''

'''
모델 성능 좋게 바꾸는 법(하이퍼 파라미터 튜닝)
1 train_size와 random state 바꾼다.
2 모델 구성시 적절한 layer 층과 적절한 노드를 사용한다.
3 훈련 횟수와 batch_size를 조절한다.
'''