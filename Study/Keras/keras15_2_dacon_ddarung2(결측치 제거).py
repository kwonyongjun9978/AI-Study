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

#결측치 처리 
#1.결측치 제거 - 데이터 10%를 지웠기 때문에 좋은 방법은 아님 
# print(train_csv.isnull().sum()) #train_csv의 컬럼별 null값 확인
train_csv = train_csv.dropna()  #결측치 제거
# print(train_csv.isnull().sum())
# print(train_csv.shape)  #(1328,10)

x=train_csv.drop(['count'], axis=1)
#print(x)   #[1328 rows x 9 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
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
model.fit(x_train, y_train, epochs=1000, batch_size=100)
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

# 평가, 예측할 때 test_csv이 아니라 train_csv를 넣어야 함. test_csv은 subsmission을 구하기 위한 x 데이터만 있고 y('count')값은 없기 때문임. 즉, test_csv는 평가하기 위한 데이터가 아님.
# 결과적으로 train은 훈련과 평가,예측를 위한 데이터이고 test는 submission 제출하기 위한 데이터임. 최종 제출은 submission만 냄.

#제출
y_submit=model.predict(test_csv)
#print(y_submit)
#print(y_submit.shape)  #(715,1)
# print(y_submit)에서 [nan]이 나오는 이유는 test.csv도 결측치가 있었다는 뜻임.
# 하지만 test.csv에서의 있는 nan은 삭제하면 안된다. submission으로 제출해야하기때문에 공란이 있으면 안된다.

#.to_csv()를 사용해서
#submission_0105.csv를 완성하시오!!

#.to_csv()는 panda 라이브러리에 있다. 우선 y_submit을 dataframe으로 만들고 그걸 .to_csv 한다.
submission['count']=y_submit # 컬럼에 채우는 방법(count란에 y_submit을 넣는다.)
# print(type(y_submit)) #<class 'numpy.ndarray'>
# print(submission)

submission.to_csv(path+'submission_0105.csv')
# print(type(submission)) #<class 'pandas.core.frame.DataFrame'>

print("걸린시간 : ", end-start)

'''
loss :  [2803.7587890625, 39.26721954345703]
RMSE :  52.95053168896345
걸린시간 :  20.955214262008667
'''

'''
왜 지금의 RMSE와 대회에 올라간 점수가 다를까?
대회에서는 내가 올린 데이터의 50%만 테스트한다. 테스트한 데이터를 public이라하고
나머지를 private이라 한 다음 맨 마지막 대회 최종 결과 발표날의 점수가 진짜 점수다.
즉, 대회에서 일부러 데이터를 왜곡한다는 의미이다.
'''




