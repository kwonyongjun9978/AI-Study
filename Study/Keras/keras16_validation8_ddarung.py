import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/ddarung/' #데이터 위치 표시
train_csv=pd.read_csv(path+'train.csv', index_col=0) #./_data/ddarung/train.csv
test_csv=pd.read_csv(path+'test.csv', index_col=0) #0번째 컬럼(id)은 데이터가 아니라 인덱스
submission=pd.read_csv(path+'submission.csv', index_col=0) #pandas의 '.read_csv' api사용

#결측치 처리 
# 1. 선형 방법을 이용하여 결측치 제거
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x=train_csv.drop(['count'], axis=1)
#print(x)   #[1328 rows x 9 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(1328,)

x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.3, shuffle=False
)

#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)
#print(submission.shape) #(715, 1)

#2.모델구성
inputs = Input(shape=(9, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32) (hidden3)
hidden5 = Dense(16) (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)

#3.컴파일, 훈련
import time 
model.compile(loss='mae', optimizer='adam')
start=time.time() 
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_data=(x_validation, y_validation))
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

submission.to_csv(path+'submission_0106.csv')

print("걸린시간 : ", end-start)

'''
loss :  2563.315673828125
RMSE :  50.62919679970939
걸린시간 :  70.55993747711182
'''

