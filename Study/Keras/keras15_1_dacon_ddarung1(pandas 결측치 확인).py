import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/ddarung/' #데이터 위치 표시
train_csv=pd.read_csv(path+'train.csv', index_col=0)         #./_data/ddarung/train.csv
test_csv=pd.read_csv(path+'test.csv', index_col=0)           # index_col=0 : 0번째 column은 index로 데이터가 아님을 명시해주는 것이다. (여기에서는 id), 항상 컴퓨터는 0부터 시작.
submission=pd.read_csv(path+'submission.csv', index_col=0)   #pandas의 '.read_csv' api사용(pandas는 데이터 분석시 사용하기 좋은 API이다.)
'''
print(train_csv) #(1459,10), count(y값)를 빼면 input_dim은 9개이다
print(train_csv.columns) #Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
                         #       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
                         #       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         #        dtype='object')
print(train_csv.info()) #결측치 확인
print(test_csv.info())  #결측치 확인
print(train_csv.describe())
'''
x=train_csv.drop(['count'], axis=1) #10개 중 count 컬럼을 제외한 나머지 9개만 inputing 
# print(x)   #[1459 rows x 9 columns]
y=train_csv['count']
# print(y)
# print(y.shape) #(1459,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=1234
)
#print(x_train.shape, x_test.shape) #(1021, 9) (438, 9)
#print(y_train.shape, y_test.shape) #(1021,) (438,)

#2.모델구성
model=Sequential()
model.add(Dense(50, input_dim=9))
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
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
# loss [nan, nan]으로 뜨는 이유: 데이터가 없는 란이 있기 때문이다. 데이터 없는 칸에 사칙연산을 해도 데이터가 없으므로 loss 자체를 구할 수 없다.
# 결측치(데이터에 값이 없는 것, null) 나쁜놈!!!
# 결측치 때문에 To Be Continue!
y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

#제출
y_submit=model.predict(test_csv)
