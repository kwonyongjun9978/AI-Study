from tensorflow.keras.datasets import cifar100
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   #(10000, 32, 32, 3) (10000, 1)

print(np.unique(y_train, return_counts=True))

#scaler
x_train, x_test = x_train / 255.0, x_test / 255.0

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D

#2. 모델
model=Sequential()
model.add(Conv2D(filters=256, kernel_size=(4,4), input_shape=(32,32,3),
                 padding='same',
                 strides=2,     
                 activation='relu')) 
model.add(MaxPooling2D())            
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', strides=2)) 
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu'))                                           
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
                                  filepath=filepath+'k35_4_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  2.767543077468872
acc :  0.3127000033855438
'''

'''
■ Stride(칸 이동)
Conv2D 에서 stride의 기본값은 1이다.
Conv2D 에서 stride를 2로 바꾸면 MaxPooling과 비슷하지만 특징만 뽑아내는 MaxPooling과는 다르게 연산량이 늘어난다.
MaxPooling의 stride의 기본값은 2이다.
'''