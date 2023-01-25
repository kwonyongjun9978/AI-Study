from tensorflow.keras.datasets import cifar100
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3) 
x_test = x_test.reshape(10000, 32*32*3)   

#scaler
x_train, x_test = x_train / 255.0, x_test / 255.0

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

#2. 모델
model=Sequential()              
model.add(Dense(256, input_shape=(32*32*3, ), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))   
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))   
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))                                     
model.add(Dense(100, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=80, 
                              restore_best_weights=True, 
                              verbose=1 )

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
                                  filepath=filepath+'k36_4_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
cnn
loss :  2.767543077468872
acc :  0.3127000033855438

dnn
loss :  4.397584915161133
acc :  0.023900000378489494
'''

'''
CNN은 DNN의 한 유형이지만 이미지 및 비디오 인식 작업에 특화되어 있으며 DNN은 다양한 작업을 수행하는 데 사용할 수 있다.
'''
