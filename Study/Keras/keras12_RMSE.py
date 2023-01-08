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
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=200, batch_size=1) #fit = 가중치(w) 생성                                  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)
# loss 값이 2개 나오는 이유: compile 해서 evalutate 할 때 loss와 metrics를 둘 다 사용했기 때문에 각각의 loss값이 나온다.
# loss :  [14.770852088928223, 2.94441294670105]

print("===================")
print(y_test)
print(y_predict)
print("===================")
 
#rmse사용하려면 sklearn api의 mean_squared_error를 import 한다음 임의로 함수를 정의해야한다
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
         # np.sqrt는 바깥쪽에 루트를 씌우는 걸 의미한다.
print("RMSE : ", RMSE(y_test, y_predict))            

#RMSE :  3.8439300378661714
#RMSE :  3.876154863400077

