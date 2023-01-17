from sklearn.datasets import fetch_california_housing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2.모델구성
# model=Sequential()
# model.add(Dense(50, input_dim=8, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(350, activation='relu'))
# model.add(Dense(400, activation='relu'))
# model.add(Dense(350, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(250, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(150, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(1))

# 2.모델구성(함수형)
input1=Input(shape=(8,))
dense1=Dense(50, activation='relu')(input1)
dense2=Dense(100, activation='relu')(dense1)
dense3=Dense(150, activation='relu')(dense2)
dense4=Dense(200, activation='relu')(dense3)
dense5=Dense(250, activation='relu')(dense4)
dense6=Dense(300, activation='relu')(dense5)
dense7=Dense(350, activation='relu')(dense6)
dense8=Dense(400, activation='relu')(dense7)
dense9=Dense(350, activation='relu')(dense8)
dense10=Dense(300, activation='relu')(dense9)
dense11=Dense(250, activation='relu')(dense10)
dense12=Dense(200, activation='relu')(dense11)
dense13=Dense(150, activation='relu')(dense12)
dense14=Dense(100, activation='relu')(dense13)
dense15=Dense(50, activation='relu')(dense14)
output1=Dense(1, activation='relu')(dense15)
model=Model(inputs=input1, outputs=output1)
model.summary() #Total params: 843,651

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=300, 
                              restore_best_weights=True, 
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=10000, batch_size=150, 
                 validation_split=0.25, callbacks=[earlyStopping], 
                 verbose=1)  

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
StandardScaler
RMSE :  0.5347818837686122
R2 :  0.7755305002454683
'''





                  



