'''
[실습]
1.train 0.7 이상
2.R2 : 0.8이상 /RMSE 사용
'''
'''
import sklearn as sk
print(sk.__version__)
'''
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1.데이터
dataset=load_boston()
x=dataset.data
y=dataset.target

'''
print(x)
print(x.shape) #(506,13)
print(y)
print(y.shape) #(506, )

print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)
'''

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)
'''
print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :',y_test)
'''
#2.모델구성
model=Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(168))
model.add(Dense(208))
model.add(Dense(248))
model.add(Dense(288))
model.add(Dense(240))
model.add(Dense(200))
model.add(Dense(160))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(9))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit(x_train, y_train, epochs=1500, batch_size=40)
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
loss :  [30.844316482543945, 3.8789093494415283]
RMSE :  5.553766069292831
R2 :  0.6271948279920736
'''

