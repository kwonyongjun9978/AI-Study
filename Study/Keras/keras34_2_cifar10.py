from tensorflow.keras.datasets import cifar10

import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000,32,32,3) (50000,)
print(x_test.shape, y_test.shape)   #(10000,32,32,3) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

#2. 모델
model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32,32,3),
                 activation='relu'))             
model.add(Conv2D(filters=64, kernel_size=(2,2))) 
model.add(Conv2D(filters=64, kernel_size=(2,2))) 
model.add(Flatten()) 
model.add(Dense(32, activation='relu'))   
                                          
model.add(Dense(10, activation='softmax'))

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
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 

ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  #filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5'
                                  filepath=filepath+'k34_2_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=100, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  2.3026187419891357
acc :  0.10000000149011612
'''