import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/ddarung/' 
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'submission.csv', index_col=0) 

#선형 방법을 이용하여 결측치 제거
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x=train_csv.drop(['count'], axis=1)
#print(x)   #[1328 rows x 9 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)

#print(x_train.shape, x_test.shape) #(929, 9) (399, 9)
#print(y_train.shape, y_test.shape) #(929,) (399,)
#print(submission.shape) #(715, 1)

#2.모델구성
model=Sequential()
model.add(Dense(50, input_dim=9))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(1))

#3.컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start=time.time() 
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.25)
end=time.time()

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

# print('x_test : ', x_test)
# print('y_predict : ', y_predict)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))              
print("RMSE : ", RMSE(y_test, y_predict)) 

#제출
y_submit=model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)  #(715,1)

#.to_csv()를 사용해서
#submission_0105.csv를 완성하시오!!

submission['count']=y_submit 
# print(type(y_submit)) #<class 'numpy.ndarray'>
# print(submission)

submission.to_csv(path+'submission_0105.csv')
# print(type(submission)) #<class 'pandas.core.frame.DataFrame'>

print("걸린시간 : ", end-start)

'''
loss :  [2416.665771484375, 38.217098236083984]
RMSE :  49.15959384045864
걸린시간 :  35.81829214096069
'''

