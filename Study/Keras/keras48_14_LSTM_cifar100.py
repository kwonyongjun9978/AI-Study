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
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(256, input_shape=(32*32*3,1), activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=80, 
                              restore_best_weights=True, 
                              verbose=1 )

model.fit(x_train, y_train, epochs=500, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  2.767543077468872
acc :  0.3127000033855438
'''