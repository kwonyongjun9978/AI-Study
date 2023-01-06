from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1.데이터
dataset=load_boston()
x=dataset.data
y=dataset.target


x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)

#2.모델구성
model=Sequential()
model.add(Dense(256, input_dim=13, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(208, activation='relu'))
model.add(Dense(248, activation='relu'))
model.add(Dense(288, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=1000, batch_size=50, validation_split=0.25)
#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
loss :  [21.916139602661133, 3.0819091796875]
RMSE :  4.681467549135127
R2 :  0.7351068073674926
'''

