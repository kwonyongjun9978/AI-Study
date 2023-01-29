import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten

a = np.array(range(1, 101))

timesteps = 5  

def split_x(dataset, timesteps):
    aaa=[]
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)
print(bbb.shape) #(96, 5)

#x는 4개, y는 1개
x = bbb[:, :-1]
y = bbb[:, -1]

print (x,y)
print (x.shape, y.shape) # (96, 4) (96,)


# 예상 y = 100~106
x_predict = np.array(range(96,106))  

x_predict = split_x(x_predict, 4) #timesteps = 4 
print(x_predict)
print(x_predict.shape) #(7, 4)

#train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,shuffle=True, random_state=444
) #train_size의 default값 : 0.75

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Conv2D는 4차원 데이터가 필요하기 때문에 (72,2,2) => (72,2,2,1) 로 reshape
x_train = x_train.reshape(72,2,2,1)
x_test = x_test.reshape(24,2,2,1)
x_predict = x_predict.reshape(7,2,2,1)

print(x_predict.shape)
print(x_train.shape)
print(x_test.shape)

#2. 모델구성
model=Sequential()
model.add(Conv2D(256, (2,1), input_shape=(2,2,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(2,1), padding='same'))  
model.add(Conv2D(filters=64, kernel_size=(2,1))) 
model.add(Flatten())
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))                                              
model.add(Dense(1, activation='relu')) 

model.summary()

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x_train,y_train,epochs=200, batch_size=8)

#4.평가,예측
loss=model.evaluate(x_test,y_test)
print('loss : ', loss)
result = model.predict(x_predict)
print('100~106예측 결과 : ', result)

'''
100~106예측 결과 :  [[100.01045 ]
 [101.01058 ]
 [102.01071 ]
 [103.01082 ]
 [104.010925]
 [105.01105 ]
 [106.011375]]
'''

