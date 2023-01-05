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

print(train_csv) #(1459,10), count(y값)를 빼면 input_dim은 9개이다
print(train_csv.shape)  #(1459,10)
print(submission.shape) #(715,1)
print(train_csv.columns) #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                         #       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                         #       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         #        dtype='object')
print(train_csv.info()) 
print(test_csv.info())
print(train_csv.describe())

#### 결측치 처리 1.제거 ####
print(train_csv.isnull().sum()) #train_csv의 null값의 합 
train_csv = train_csv.dropna() #결측치 제거
print(train_csv.isnull().sum())
print(train_csv.shape)

x=train_csv.drop(['count'], axis=1)
print(x)   #[1459 rows x 9 columns]
y=train_csv['count']
print(y)
print(y.shape) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.9,
    shuffle=True,
    random_state=1234
)
print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
print(y_train.shape, y_test.shape) #(929,) (399,)

#2.모델구성
model=Sequential()
model.add(Dense(44, input_dim=9))
model.add(Dense(204))
model.add(Dense(169))
model.add(Dense(138))
model.add(Dense(98))
model.add(Dense(66))
model.add(Dense(49))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
model.fit(x_train, y_train, epochs=300, batch_size=89)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict=model.predict(x_test)
print('x_test : ', x_test)
print('y_predict : ', y_predict)

def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

#제출
y_submit=model.predict(test_csv)
print(y_submit)
print(y_submit.shape)  #(715,1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성하시오!!

print(submission)
submission['count']=y_submit
print(submission)

submission.to_csv(path+'submission_0105.csv')
print("RMSE : ", RMSE(y_test, y_predict)) 


