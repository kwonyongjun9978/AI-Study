import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape)  #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)    #(10000, 28, 28) (10000,)

# import matplotlib.pyplot as plt
# plt.imshow(x_train[10], 'gray')
# plt.show()

x_train=x_train.reshape(60000,28,28,1) 
x_test=x_test.reshape(10000,28,28,1)   

print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28, 1) (10000,)

print(np.unique(y_train, return_counts=True))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

#2. 모델
model=Sequential()
model.add(Conv2D(filters=128, kernel_size=(3,3), input_shape=(28,28,1),
                 padding='same', #valid
                 activation='relu'))                                  
model.add(MaxPooling2D())                      
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same'))  
model.add(MaxPooling2D())
model.add(Conv2D(filters=32, kernel_size=(2,2))) 
model.add(Flatten()) 
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
                                  filepath=filepath+'k35_2_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=1000, batch_size=32,
          validation_split=0.2,
          callbacks=[earlyStopping, ModelCheckpoint],
          verbose=1)

#4.평가, 예측
results=model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('acc : ', results[1])

'''
loss :  0.324116587638855
acc :  0.8873000144958496
'''