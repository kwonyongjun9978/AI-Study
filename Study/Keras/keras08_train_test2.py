import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10]) #(10, ) cf)([[1,2,3,4,5,6,7,8,9,10]]) (10,1)
y=np.array(range(10))              #(10, )
#y=wx+b //w=1,b=-1


#슬라이싱 사용
X_train=x[:-3] #[0:7]    [:7]
X_test=x[-3:]  #[7:10]   [7:]
Y_train=y[:-3]
Y_test=y[-3:]

'''

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
loss :  0.01052890531718731
[11]의 결과 [[10.005239]]
'''



