import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

#실습
#슬라이싱
x_train=x[1:11] 
y_train=y[1:11]
x_test=x[11:14]  
y_test=x[11:14]  
x_val=x[14:17]
y_val=y[14:17]

'''
x_train=np.array(range(1,11)) #(10,) 훈련데이터
y_train=np.array(range(1,11))
x_test=np.array([11,12,13]) #(1,3) 평가데이터
y_test=np.array([11,12,13])
x_validation=np.array([14,15,16]) #(1,3) 검증데이터
y_validation=np.array([14,15,16])
'''
#2.모델
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=1,
          validation_data=(x_val, y_val))

#4.평가, 예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)

result=model.predict([17])
print("17의 예측값 : ", result)