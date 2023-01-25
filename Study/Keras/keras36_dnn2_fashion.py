import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    #(10000, 28, 28) (10000,)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[10], 'gray')
# plt.show()

x_train = x_train.reshape(60000, 28*28) #(60000, 28, 28) (60000,)
x_test = x_test.reshape(10000, 28*28)   #(10000, 28, 28) (10000,)

#scaler
x_train, x_test = x_train / 255.0, x_test / 255.0

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

#2. 모델
model=Sequential()             
model.add(Dense(128, input_shape=(784, ), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))   
model.add(Dropout(0.3))
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
                                  filepath=filepath+'k36_2_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
cnn
loss :  0.324116587638855
acc :  0.8873000144958496
'''

'''
dnn
loss :  0.3319469094276428
acc :  0.8866999745368958
'''