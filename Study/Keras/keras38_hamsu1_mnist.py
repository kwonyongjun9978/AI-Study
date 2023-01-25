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

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input

#2. 모델
# model=Sequential()
# model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
#                  padding='same', #valid
#                  activation='relu'))                              #(28,28,128)    
# model.add(MaxPooling2D())  #연산량이 줄어든다                       (14,14,128)
# model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))  #(14,14,64)
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=32, kernel_size=(2,2))) 
# model.add(Flatten()) 
# model.add(Dense(32, activation='relu'))                                            
# model.add(Dense(10, activation='softmax'))

#함수형 모델
A1=Input(shape=(28,28,1)) 
B0=Conv2D(filters=128, kernel_size=(3,3), padding='same', #valid
                 activation='relu')(A1)
B1=MaxPooling2D(2,2)(B0)
B2=Conv2D(filters=64, kernel_size=(2,2), padding='same')(B1)
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
                                  filepath=filepath+'k38_1_'+date+'_'+filename)

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
loss :  0.05545061081647873
acc :  0.9832000136375427
'''


