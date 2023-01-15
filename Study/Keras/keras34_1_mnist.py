import numpy as np
from tensorflow.keras.datasets import mnist


#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (뒤에 1이 없으니까 흑백) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

#60000장의 데이터 셋을 가지고 훈련을 한다

x_train=x_train.reshape(60000,28,28,1) 
x_test=x_test.reshape(10000,28,28,1)   

print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28,28,1),
                 activation='relu'))             #(27,27,128)
model.add(Conv2D(filters=64, kernel_size=(2,2))) #(26,26,64)
model.add(Conv2D(filters=64, kernel_size=(2,2))) #(25,25,64)
model.add(Flatten()) #40000
model.add(Dense(32, activation='relu'))   #input_shape=(40000,)
                                          #(60000,40000)=(batch_size,input_dim)
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=10, 
                              restore_best_weights=True, 
                              verbose=1 )

import datetime
date = datetime.datetime.now() #현재 시간 반환
print(date) 
print(type(date)) #<class 'datetime.datetime'>
date=date.strftime("%m%d_%H%M") #문자열 타입으로 변환
print(date) 
print(type(date)) #<class 'str'>

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' #0037-0.0048.hdf8 

ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  #filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5'
                                  filepath=filepath+'k34_1_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
CNN에 넣으려면 독립변수 데이터를 4차원 텐서로 reshape 해야한다.
reshape 할 때는 원하는 모양을 적으면 된다.
스칼라 (3, )를 벡터 (3,1)로 바꿀 때에는 .reshape(3,1) 또는 .reshape(-1,1)로 표현 가능했지만,
텐서로 바꿀 때에는 -1이 안 통한다. 그냥 다 적어주는 게 좋다.
one-hot 안 해줬으므로 categorical이 아니라 sparse_categorical 쓴다.
Dense일 때는 CPU 쓰면 더 좋았지만 CNN일 때는 GPU 쓰는 게 더 빠르다.
'''

'''
loss :  0.11643264442682266
acc :  0.970300018787384
'''
