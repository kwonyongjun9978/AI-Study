import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

a = np.array(range(1, 101))

#예상 y = 100, 107

timesteps = 5  #x는 4개, y는 1개

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape) #(96, 5)

x=bbb[:, :-1]
y=bbb[:, -1] 
print(x,y)

print(x.shape, y.shape) #(96, 4) (96,)

x = x.reshape(96,4,1)


x_predict = np.array(range(96, 106))

timesteps = 4  

def split_y(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

ddd = split_y(x_predict, timesteps)
print(ddd)
print(ddd.shape) #(7, 4)

ddd = ddd.reshape(7,4,1)

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

model.fit(x,y,epochs=100, batch_size=2)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
x_predict=np.array(ddd)
result = model.predict(x_predict)
print('결과 : ', result)