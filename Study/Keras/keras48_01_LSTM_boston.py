from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset=load_boston()
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

print(x_train.shape, x_test.shape) #(404, 13) (102, 13)

x_train = x_train.reshape(404,13,1)
x_test = x_test.reshape(102,13,1)

print(x_train.shape, x_test.shape)

#2.모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(13,1), activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=200, 
                              restore_best_weights=True, 
                              verbose=2)  
               
hist = model.fit(x_train, y_train, epochs=1000, batch_size=2, 
                 validation_split=0.2, callbacks=[earlyStopping], 
                 verbose=2)  

#4.평가,예측
mse=model.evaluate(x_test, y_test) 
print('mse : ', mse)

from sklearn.metrics import mean_squared_error, r2_score
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
dnn
mse :  [19.34838104248047, 2.5393197536468506]
R2 :  0.8027265452002512

cnn
mse :  [12.695144653320312, 2.433328866958618]
R2 :  0.8705620076953748

LSTM
mse :  [16.38949966430664, 2.654872179031372]
R2 :  0.8328948571791543
'''








                  



