import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/bike/' #데이터 위치 표시
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

#결측치 처리 
train_csv = train_csv.dropna()  

x=train_csv.drop(['casual','registered','count'], axis=1)
#print(x) #[10886 rows x 8 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=123
)

print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)
print(submission.shape) 

#2.모델구성
'''
활성화함수(activation)
=레이어에서 레이어로 이동할때 값자체를 한정시킴
sigmoid : 0~1
relu : 0이하는 0, 0이상은 그 값 그대로 한정된다
'''
model=Sequential()
model.add(Dense(256, input_dim=8, activation='relu')) 
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='linear'))

#3.컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam')
start=time.time() 
model.fit(x_train, y_train, epochs=690, batch_size=204)
end=time.time()

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict=model.predict(x_test)

#print('x_test : ', x_test)
#print('y_predict : ', y_predict)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))              
print("RMSE : ", RMSE(y_test, y_predict)) 

#제출
y_submit=model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)  #(715,1)


#.to_csv()를 사용해서
#submission_0106.csv를 완성하시오!!

submission['count']=y_submit
#print(submission)

submission.to_csv(path+'submission_010601.csv')

print("걸린시간 : ", end-start)


'''
loss :  23266.326171875
RMSE :  152.5330470652978
'''
