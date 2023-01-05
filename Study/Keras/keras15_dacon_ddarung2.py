import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/ddarung/' #데이터 위치 표시
train_csv=pd.read_csv(path+'train.csv', index_col=0) #./_data/ddarung/train.csv
test_csv=pd.read_csv(path+'test.csv', index_col=0) #0번째 컬럼(id)은 데이터가 아니라 인덱스
submission=pd.read_csv(path+'submission.csv', index_col=0) #pandas의 '.read_csv' api사용

#결측치 처리 
#1.결측치 제거 - 데이터 10%를 지웠기 때문에 좋은 방법은 아님 
#print(train_csv.isnull().sum()) #train_csv의 컬럼별 null값
train_csv = train_csv.dropna()  #결측치 제거
#print(train_csv.isnull().sum())
#print(train_csv.shape)  #(1328,10)

x=train_csv.drop(['count'], axis=1)
#print(x)   #[1328 rows x 9 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=44
)

#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)
#print(submission.shape) #(715, 1)

#2.모델구성
model=Sequential()
model.add(Dense(256, input_dim=9))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam')
start=time.time() 
model.fit(x_train, y_train, epochs=10, batch_size=32)
end=time.time()

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict=model.predict(x_test)

#print('x_test : ', x_test)
#print('y_predict : ', y_predict)

def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))              
print("RMSE : ", RMSE(y_test, y_predict)) 

#제출
y_submit=model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)  #(715,1)


#.to_csv()를 사용해서
#submission_0105.csv를 완성하시오!!

submission['count']=y_submit
#print(submission)

submission.to_csv(path+'submission_0105.csv')

print("걸린시간 : ", end-start)

#걸린시간 :  9.419771671295166
