from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset=fetch_california_housing()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)

#Scaler(데이터 전처리) 
#scaler = StandardScaler()
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(16512, 8) (4128, 8)

x_train = x_train.reshape(16512,8,1,1)
x_test = x_test.reshape(4128,8,1,1)

# 2.모델구성(함수형)
model=Sequential()
model.add(Conv2D(512, (4,1), input_shape=(8,1,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(2,1), padding='same'))  
model.add(Conv2D(filters=128, kernel_size=(2,1))) 
model.add(Conv2D(filters=64, kernel_size=(2,1))) 
model.add(Flatten())
model.add(Dense(256, activation='relu')) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))                                              
model.add(Dense(1, activation='relu')) 
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=500, 
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

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=2,
                                  save_best_only=True,
                                  filepath=filepath+'k39_2_'+date+'_'+filename)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=200, 
                 validation_split=0.2, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=2)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
dnn
loss :  0.2827185094356537
RMSE :  0.5317127633608937
R2 :  0.7780995739992294

cnn
loss :  0.2449636459350586
RMSE :  0.49493800838915014
R2 :  0.8077326333581126
'''
                  



