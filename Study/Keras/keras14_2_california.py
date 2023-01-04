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
    train_size=0.7,
    shuffle=True,
    random_state=123
)
#2.모델구성
model=Sequential()
model.add(Dense(50, input_dim=8))
model.add(Dense(100))
model.add(Dense(150))
model.add(Dense(200))
model.add(Dense(250))
model.add(Dense(300))
model.add(Dense(350))
model.add(Dense(400))
model.add(Dense(350))
model.add(Dense(300))
model.add(Dense(250))
model.add(Dense(200))
model.add(Dense(150))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=1000, batch_size=100)
#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

'''
print("===================")
print(y_test)
print(y_predict)
print("===================")
'''
#R2=정확도와 비슷한 개념,1에 가까울수록 좋다
#R2를 사용하려면 sklearn을 import 한다음 임의로 함수를 정의해야한다
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
loss :  [0.6051779985427856, 0.5777385234832764]
RMSE :  0.7779319001592162
R2 :  0.5423254670119855
'''