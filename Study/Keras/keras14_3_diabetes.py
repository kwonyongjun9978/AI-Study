'''
실습
R2 0.62 이상
'''
from  sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

'''
print(x)
print(x.shape) #(442, 10)
print(y)
print(y.shape) #(442,)

print(datasets.feature_names)
print(datasets.DESCR)
'''

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)

#2.모델구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=600, batch_size=40)

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
loss :  [2723.51904296875, 43.25250244140625]
RMSE :  52.18734413029384
R2 :  0.5677069015608767
'''

