import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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
'''
feature값을 2로 변경([batch, timesteps, feature])
■ feature를 2로 바꾸면 좋은 점?
feature를 늘리면 자료에서 더 복잡한 관계와 패턴을 찾을 수 있어서 모델의 성능과 정확도가 올라가는 장점이 있지만,
연산량이 늘어나 과적합될 수 있을 뿐만 아니라 훈련 시간이 늘어난다는 단점이 있다.
=> 가장 좋은 균형을 찾는 것이 중요함.
'''
x_train = x_train.reshape(72,2,2)
x_test = x_test.reshape(24,2,2)
x_predict = x_predict.reshape(7,2,2)

print(x_predict.shape)
print(x_train.shape)
print(x_test.shape)

#2. 모델구성
model = Sequential()
model.add(LSTM(512, input_shape=(2,2), activation='relu')) 
model.add(Dense(256, activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
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
loss :  0.0007610560278408229
100~106예측 결과 :  [[100.04496 ]
 [101.04579 ]
 [102.046715]
 [103.04763 ]
 [104.048615]
 [105.049614]
 [106.0507  ]]
'''
