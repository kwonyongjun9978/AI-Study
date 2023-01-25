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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D, Input

#2. 모델
# model=Sequential()
# model.add(Conv2D(filters=256, kernel_size=(4,4), input_shape=(32,32,3),
#                  padding='same',
#                  strides=2,     #stride 기본값 = 1, maxpooling에서는 stride 기본값이 = 2
#                  activation='relu')) 
# model.add(MaxPooling2D())            
# model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', strides=2)) 
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding='same')) 
# model.add(Flatten()) 
# model.add(Dense(128, activation='relu'))                                           
# model.add(Dense(100, activation='softmax'))

#함수형 모델
A1=Input(shape=(32,32,3)) 
B0=Conv2D(filters=256, kernel_size=(4,4), padding='same', #valid
                 activation='relu')(A1)
B1=MaxPooling2D(2,2)(B0)
B2=Conv2D(filters=128, kernel_size=(3,3), padding='same')(B1)
B3=MaxPooling2D(2,2)(B2)
B4=Conv2D(filters=64, kernel_size=(2,2), padding='same')(B3)
B5=(Flatten())(B4)
B6=Dense(128, activation='relu')(B5)
B7=Dense(100, activation='softmax')(B6)
model=Model(inputs=A1, outputs=B7)

model.summary()

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
                                  filepath=filepath+'k38_4_'+date+'_'+filename)

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
