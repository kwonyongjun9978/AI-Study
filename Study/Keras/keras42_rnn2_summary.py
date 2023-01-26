import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) #(10, )

#시계열 데이터는 y 가 없다

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])
y = np.array([4,5,6,7,8,9,10])
print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(7,3,1)    # -> [[[1],[2],[3]],
                        #    [[2],[3],[4]], ...]
print(x.shape) #(7, 3, 1) 3차원

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(units =256, activation='relu', input_shape=(3,1)))
                                                   #(N, 3, 1) -> ([batch, timesteps, feature])
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()

# Param 개수
# 256 * (256 + 1 + 1) = 66048
# units * (feature + bias + units)