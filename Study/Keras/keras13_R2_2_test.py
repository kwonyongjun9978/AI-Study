#실습
#R2를 음수가 아닌 0.5이하로 줄이기(즉 R2를 강제적으로 나쁘게 만들기)
#1.데이터는 건들지 말것
#2.레이어는 인풋 아웃풋 포함 7개 이상
#3.batch_size=1
#4.히든레이어의 노드는 각각 10개 이상 100개 이하
#5. train 70%
#6. epochs 100번 이상
#7. loss지표는 mse 또는 mae
#8. activation 사용 금지

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1.데이터
x=np.array(range(1,21))
y=np.array(range(1,21))

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=123
)

#2.모델구성
model=Sequential()
model.add(Dense(400, input_dim=1))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(200))
model.add(Dense(10))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=100, batch_size=1) #fit = 가중치(w) 생성                                  

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
'''
def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
            #루트                  mse                    =rmse
print("RMSE : ", RMSE(y_test, y_predict))            
'''
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)


