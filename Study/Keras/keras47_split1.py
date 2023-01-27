import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape) #(6, 5)

x=bbb[:, :-1]
y=bbb[:, -1] #y=bbb[:, 5]
print(x,y)
'''
[[1 2 3 4]
 [2 3 4 5]
 [3 4 5 6]
 [4 5 6 7]
 [5 6 7 8]
 [6 7 8 9]] [ 5  6  7  8  9 10]
'''
print(x.shape, y.shape) #(6, 4) (6,)

x = x.reshape(6,4,1)

#2. 모델구성
model = Sequential()
model.add(LSTM(512, input_shape=(4,1), activation='relu')) 
model.add(Dense(256, activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x,y,epochs=1500, batch_size=2)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
x_predict=np.array([7,8,9,10]).reshape(1,4,1)
result = model.predict(x_predict)
print('[7,8,9,10]의 결과 : ', result)

'''
loss :  1.689546297711786e-05
[7,8,9,10]의 결과 :  [[11.006391]]
'''