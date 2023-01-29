from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
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

x_train = x_train.reshape(16512,8,1)
x_test = x_test.reshape(4128,8,1)

# 2.모델구성
model = Sequential()
model.add(Conv1D(512, 2, input_shape=(8,1), activation='relu')) 
model.add(Dense(256, activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=2, 
                              restore_best_weights=True, 
                              verbose=2)

hist = model.fit(x_train, y_train, epochs=10, batch_size=200, 
                 validation_split=0.2, callbacks=[earlyStopping], 
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
                  



