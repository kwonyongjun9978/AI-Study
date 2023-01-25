from tensorflow.keras.datasets import cifar10

import numpy as np

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000,32,32,3) (50000,)
print(x_test.shape, y_test.shape)   #(10000,32,32,3) (10000,)

print(np.unique(y_train, return_counts=True))

#scaler
x_train, x_test = x_train / 255.0, x_test / 255.0

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input

#2. 모델
# model=Sequential()
# model.add(Conv2D(filters=128, kernel_size=(4,4), input_shape=(32,32,3),
#                  padding='same',
#                  activation='relu')) 
# model.add(MaxPooling2D())            
# model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')) 
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu')) 
# model.add(Flatten()) 
# model.add(Dense(16, activation='relu'))                                   
# model.add(Dense(10, activation='softmax'))

#함수형 모델
A1=Input(shape=(32,32,3)) 
B0=Conv2D(filters=128, kernel_size=(4,4), padding='same', #valid
                 activation='relu')(A1)
B1=MaxPooling2D(2,2)(B0)
B2=Conv2D(filters=64, kernel_size=(3,3), padding='same')(B1)
B3=MaxPooling2D(2,2)(B2)
B4=Conv2D(filters=32, kernel_size=(2,2), padding='same')(B3)
B5=(Flatten())(B4)
B6=Dense(32, activation='relu')(B5)
B7=Dense(10, activation='softmax')(B6)
model=Model(inputs=A1, outputs=B7)

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=30, 
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
                                  filepath=filepath+'k38_3_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  0.9729945063591003
acc :  0.6776999831199646
'''