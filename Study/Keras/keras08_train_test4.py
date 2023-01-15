import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([range(10), range(21,31), range(201,211)])
y=np.array([[1,2,3,4,5,6,7,8,9,10], [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
x=x.T #(10,3)
y=y.T #(10,2)

#[실습] train_test_split를 이용하여 7:3으로 잘라서 모델 구현
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(   
    x,y,                 
    train_size=0.7,       
    #test_size=0.3,      
    #shuffle=True,       
    random_state=123     
)

print('X_train :', X_train)
print('X_test :', X_test)
print('Y_train :', Y_train)
print('Y_test :',Y_test)

#2.모델구성
model=Sequential() 
model.add(Dense(10, input_dim=3)) 
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))

#3.컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)

#4.평가,예측
loss=model.evaluate(X_test,Y_test)
print('loss : ', loss)
results=model.predict([[9, 30, 210]])
print('[10, 1.4, 0]의 예측값 : ', results)

