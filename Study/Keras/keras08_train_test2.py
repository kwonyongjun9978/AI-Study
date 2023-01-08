import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10]) #(10, ) 
y=np.array(range(10))              #(10, )

#슬라이싱 사용
x_train = x[:7] # 1,2,3,4,5,6,7
x_test = x[7:]  # 8,9,10
y_train = y[:7] # 0,1,2,3,4,5,6
y_test = y[7:]  # 7,8,9 

'''
X_train=x[:-3] 
X_test=x[-3:]  
Y_train=y[:-3]  
Y_test=y[-3:]  
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
model.fit(x_train,y_train,epochs=200, batch_size=1)

#4.평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)
result=model.predict([11])
print('[11]의 결과', result)

'''
loss :  0.01052890531718731
[11]의 결과 [[10.005239]]
'''



