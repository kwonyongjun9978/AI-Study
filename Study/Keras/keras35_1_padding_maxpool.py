import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

x_train=x_train.reshape(60000,28,28,1) 
x_test=x_test.reshape(10000,28,28,1)   

print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#2. 모델
model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
                 padding='same', 
                 activation='relu'))                              #(28,28,128)    
model.add(MaxPooling2D())  #연산량이 줄어든다                       (14,14,128)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))  #(14,14,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(2,2))) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))                                            
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=50, 
                              restore_best_weights=True, 
                              verbose=2)

import datetime
date = datetime.datetime.now() 
print(date) 
print(type(date)) 
date=date.strftime("%m%d_%H%M") 
print(date) 
print(type(date)) 

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  

ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  filepath=filepath+'k35_1_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
기존성능
loss :  0.11296992003917694
acc :  0.9707000255584717

padding적용
loss :  0.09387045353651047
acc :  0.97079998254776

padding+MaxPooling적용
loss :  0.06569646298885345
acc :  0.9819999933242798
'''

'''
■ padding 원리
convolution 과정에서 반드시 padding이 필요하다.
convolution filter를 통과하면 input 이미지가 작아지는데, 이때 padding을 사용하면 그대로 유지할 수 있다.
이미지의 외곽을 빙 둘러서 1픽셀씩 더 크게 만들고 추가된 1 픽셀에 0의 값을 주며 이를 zero padding이라고 함.
즉 convolution에는 크게 2가지 종류가 있다.
1 Valid convolution(default) : padding 없음
이미지 사이즈 변화 n x m * f x q => n-f+1 x m-q+1
2 Same convolution : 아웃풋 이미지의 사이즈 동일하게 padding
n+2p-f+1 x m+2t-q+1 = n x m 이려면
p=(f-1)/2
t=(q-1)/2
# convolution에서 padding을 하는 이유
1 아웃풋 이미지의 크기를 유지하기 위해
2 Edge 쪽 픽셀 정보를 더 잘 이용하기 위해

■ padding 하는 방법
모델 안에 Conv2D 안에 padding이 있음. Conv2D 속괄호 안에 있어야 함.
Conv2D(filter, kernel_size, input_shape, padding, activation)

■ Pooling 원리, 목적
CNN에서 pooling이란 연산 없이 특징만 뽑아내는 과정이다.
1 Max Pooling : 정해진 크기 안에서 가장 큰 값만 뽑아낸다.
2 Average Pooling : 정해진 크기 안에서 값들의 평균을 뽑아낸다.

■ MaxPooling 하는 방법
from tensorflow.keras.layers import MaxPooling2D
model.add(MaxPooling2D())
- pool_size : pooling에 사용할 filter의 크기를 정하는 것.(단순한 정수, 또는 튜플형태 (N,N))
- strides : pooling에 사용할 filter의 strides를 정하는 것.
- padding : "valide"(=padding을 안하는것) or "same"(=pooling결과 size가 input size와 동일하게 padding)
'''

