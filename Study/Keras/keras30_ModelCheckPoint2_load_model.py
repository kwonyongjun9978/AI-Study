from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
path='./_save/'

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
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

'''
# 2.모델구성(함수형)
input1=Input(shape=(13,))
dense1=Dense(256, activation='relu')(input1)
dense2=Dense(128, activation='relu')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
output1=Dense(1, activation='relu')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary() #Total params: 47,361

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=40, 
                              restore_best_weights=True, 
                              verbose=1 )  

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  filepath=path+'MCP/keras30_ModelCheckPoint1.hdf5')
                                
                                                                      
hist = model.fit(x_train, y_train, epochs=5000, batch_size=2, 
                 validation_split=0.2, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=1)  
'''
#ModelCheckpoint load
model=load_model(path+'MCP/keras30_ModelCheckPoint1.hdf5')

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

#MCP 불러오기  R2 :  0.8757832219281168

#ModelCheckPoint를 로드하면 세이브했던 결과와 동일하다.







                  



