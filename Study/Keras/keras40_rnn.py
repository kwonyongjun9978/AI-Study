import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. 데이터(원래 시계열 데이터는 y 가 없다)
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) #(10, )

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
model.add(SimpleRNN(256, activation='relu', input_shape=(3,1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x,y,epochs=1500, batch_size=2)

#4.평가,예측
loss=model.evaluate(x,y)
print('loss : ', loss)
y_pred = np.array([8,9,10]).reshape(1,3,1) #predict에서의 input_shape도 model.add에 들어가는 input_shape와와 동일해야하며 행무시 열우선이므로 앞에 1 추가해야한다.
result = model.predict(y_pred)
print('[8,9,10의 결과 : ', result)

'''
loss :  8.091923575648252e-08
[8,9,10의 결과 :  [[10.998638]]
'''

'''
■ RNN (Recurrent Neural Network)
순환 신경망은 시계열 또는 자연어와 같은 순차적 데이터를 처리할 수 있는 신경망의 한 유형이다.
RNN은 이전 입력에 대한 정보를 유지할 수 있는 메모리가 있다.
즉, RNN은 상태를 계산할 때 이전 상태를 사용하는 피드백 루프를 네트워크에 도입함으로써 작동한다.

DNN은 1 2 3 4 5 6 X 에서 1~6 까지 계산해 y=x 함수 그려 X 예측함
RNN은 1 2 3 4 5 6 X 에서
  X     Y
1 2 3 | 4	=> 1에서2, 2에서3, 3에서4 찾음
2 3 4 | 5
3 4 5 | 6
4 5 6 | X
Y1    Y2    Y3
H1 -> H2 -> H3
W1    W2    W3     
X1 -> X2 -> X3
H1 = X1W1 + B
H2 = X2W2 + X1W1 + B
H3 = X3W3 + X2W2 + X1W1 + B
이런 식으로 계산함
다음 뉴런으로 넘겨줄 때 tan 함수로 처리하기 때문에 값이 많이 커지진 않는다.
'''