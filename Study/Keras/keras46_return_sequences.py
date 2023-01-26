import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. 데이터
x=np.array([[1,2,3],[2,3,4],[3,4,5],
            [4,5,6],[5,6,7],[6,7,8],
            [7,8,9],[8,9,10],[9,10,11],
            [10,11,12],[20,30,40],
            [30,40,50],[40,50,60]])
y=np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) #(13, 3) (13,)

x = x.reshape(13,3,1)

#2. 모델구성
model = Sequential()          #(N , 3, 1)
model.add(LSTM(256, input_shape=(3,1), activation='relu', 
               return_sequences=True))   # (N, 3, 256)
model.add(LSTM(128, activation='relu'))  # (N, 128)                                                                                              
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()
'''
■ 연속으로 LSTM 구성할 때 reshape 하는 방법
LSTM은 3차원 자료 줘야하는데 LSTM 거치면 2차원 자료로 바뀌므로 LSTM 연속으로 layer 추가하고 싶으면 다시 reshape 해야 한다.
reshape 하는 방법에는 2가지가 있는데, reshape layer를 추가하거나 이전 LSTM에서 return_sequence=True로 설정하면 된다.
'''
#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x,y,epochs=1500, batch_size=2)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
x_predict=np.array([50,60,70]).reshape(1,3,1)
result = model.predict(x_predict)
print('[50,60,70]의 결과 : ', result)

'''
loss :  0.09062328934669495
[50,60,70]의 결과 :  [[80.202965]]
'''