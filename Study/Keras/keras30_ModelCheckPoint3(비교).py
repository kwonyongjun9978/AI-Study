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
    random_state=123
)

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


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
                              patience=50, 
                              restore_best_weights=False, #기본값:False 
                              verbose=1 )  

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  filepath=path+'MCP/keras30_ModelCheckPoint3.hdf5')
                                
                                                                      
hist = model.fit(x_train, y_train, epochs=5000, batch_size=2, 
                 validation_split=0.25, callbacks=[earlyStopping, ModelCheckpoint], 
                 verbose=1)  

#모델+가중치save
model.save(path+'keras_30_ModelCheckPoint3_save_model.h5')

#4.평가,예측
print("=========================1. 기본 출력===========================")
mse=model.evaluate(x_test, y_test) 
print('mse : ', mse)

from sklearn.metrics import mean_squared_error, r2_score
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

print("=======================2. load_model 출력========================")
model2=load_model(path + 'keras_30_ModelCheckPoint3_save_model.h5')
mse=model2.evaluate(x_test, y_test) 
print('mse : ', mse)

y_predict=model2.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

print("====================3. ModelCheckPoint 출력======================")
model3=load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse=model3.evaluate(x_test, y_test) 
print('mse : ', mse)

y_predict=model3.predict(x_test)
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
이론상으로 restore_best_weights=False(Default) 하면 ModelCheckPoint가 기본출력이랑 load_model한 것보다 더 좋게 나온다.
왜냐하면, restore_best_weights=False(Default) 했을 때, ModelCheckPoint는 오차가 최소인 지점에서 브레이크 건 상태에서 결과값을 내 모델과 가중치를 저장하고,
그냥 .save는 오차가 최소인 지점에서 브레이크를 걸지 않고 patience 만큼 더 간 상태에서 결과값을 내 모델과 가중치를 저장하기 때문이다.
만약 restore_best_weights=True로 설정해주면 ModelCheckPoint랑 .save 둘 다 최소인 지점에서 브레이크를 걸기 때문에 동일한 값이 나온다.
가끔 restore_best_weights=False 해도 ModelCheckPoint의 결과가 안 좋을 때에도 있다.
그 이유는 평가하는 데이터가 x_train이 아니라 x_test이기 때문이다.
x_train 에서는 브레이크 걸었을 때 가중치가 안 걸었을 때보다 더 좋게 나오지만 x_test 에서는 브레이크 걸었을 때 가중치가 안 걸었을 때보다 안 좋을 수 있기 때문이다.
'''








                  



