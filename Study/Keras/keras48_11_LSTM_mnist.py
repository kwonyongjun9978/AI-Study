import numpy as np
from tensorflow.keras.datasets import mnist

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (뒤에 1이 없으니까 흑백) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. 모델
model = Sequential()
model.add(LSTM(256, input_shape=(28*28,1), activation='relu'))                                                                                                
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
                              patience=50, 
                              restore_best_weights=True, 
                              verbose=2)

model.fit(x_train, y_train, epochs=300, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  0.11296992003917694
acc :  0.9707000255584717
'''
