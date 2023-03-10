import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path='./_data/bike/' 
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

train_csv = train_csv.dropna()  #결측치 제거

x=train_csv.drop(['casual','registered','count'], axis=1)
#print(x) #[10886 rows x 8 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=44
)

print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)
print(submission.shape) 

#2.모델구성
model=Sequential()
model.add(Dense(32, input_dim=8, activation='relu')) 
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
import time 
model.compile(loss='mse', optimizer='adam')
start=time.time() 
model.fit(x_train, y_train, epochs=100, batch_size=10,
          validation_split=0.2)
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

submission.to_csv(path+'submission_010803.csv')

print("걸린시간 : ", end-start)


'''
010801
loss :  22147.0625
RMSE :  148.81888841142472
걸린시간 :  123.88783955574036

010801
loss :  22387.5859375
RMSE :  149.62481718191893
걸린시간 :  139.6375584602356

010802
loss :  22255.8671875
RMSE :  149.18398443418045
걸린시간 :  178.96774125099182

010803
loss :  22086.767578125
RMSE :  148.61617584960914
걸린시간 :  62.70538115501404
'''
