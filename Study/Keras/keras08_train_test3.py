import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10]) #(10, ) cf)([[1,2,3,4,5,6,7,8,9,10]]) (10,1)
y=np.array(range(10))              #(10, )
#y=wx+b //w=1,b=-1

#슬라이싱 사용
# X_train=x[:-3] #[0:7]    [:7]
# X_test=x[-3:]  #[7:10]   [7:]
# Y_train=y[:-3]
# Y_test=y[-3:]



#사이킷런 사용
'''
훈련데이터와 평가데이터를 분리할때 전체데이터 내에서 훈련데이터와 평가데이터를 분리한다(과적합 문제가 발생할수도 있음)
|----------|
    |----------|
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(   # ()=함수
    x,y,                 #파라미터(x와y에 값을 대입)
    train_size=0.7,      #파라미터 
    #test_size=0.3,      #파라미터
    #shuffle=True,       #파라미터(shuffle=랜덤하게 섞는다),기본값(default)=true
    random_state=123     #파라미터
)

'''
print('X_train :', X_train)
print('X_test :', X_test)
print('Y_train :', Y_train)
print('Y_test :',Y_test)
'''

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))


#3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(X_train,Y_train,epochs=200, batch_size=1)

#4.평가, 예측
loss=model.evaluate(X_test,Y_test)
print('loss : ', loss)
result=model.predict([11])
print('[11]의 결과', result)

'''
loss :  0.045503169298172
[11]의 결과 [[9.876015]]
'''



