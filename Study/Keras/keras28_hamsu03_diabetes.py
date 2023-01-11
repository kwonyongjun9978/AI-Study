from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset=load_diabetes()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)
# scaler = StandardScaler()
scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2.모델구성
# model=Sequential()
# model.add(Dense(88, input_dim=10, activation='relu'))
# model.add(Dense(168, activation='relu'))
# model.add(Dense(208, activation='relu'))
# model.add(Dense(248, activation='relu'))
# model.add(Dense(288, activation='relu'))
# model.add(Dense(240, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(160, activation='relu'))
# model.add(Dense(120, activation='relu'))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(9, activation='relu'))
# model.add(Dense(1))

# 2.모델구성(함수형)
input1=Input(shape=(10,))
dense1=Dense(88, activation='relu')(input1)
dense2=Dense(168, activation='relu')(dense1)
dense3=Dense(208, activation='relu')(dense2)
dense4=Dense(248, activation='relu')(dense3)
dense5=Dense(288, activation='relu')(dense4)
dense6=Dense(240, activation='relu')(dense5)
dense7=Dense(200, activation='relu')(dense6)
dense8=Dense(160, activation='relu')(dense7)
dense9=Dense(120, activation='relu')(dense8)
dense10=Dense(80, activation='relu')(dense9)
dense11=Dense(40, activation='relu')(dense10)
dense12=Dense(20, activation='relu')(dense11)
dense13=Dense(9, activation='relu')(dense12)
output1=Dense(1, activation='relu')(dense13)
model=Model(inputs=input1, outputs=output1)
model.summary() 

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=70, 
                              restore_best_weights=True, 
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=600, batch_size=20, 
                 validation_split=0.25, callbacks=[earlyStopping], 
                 verbose=1)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("===============================")
print(hist) #
print("===============================")
print(hist.history) 
print("===============================")
print(hist.history['loss'])   
print("===============================")
print(hist.history['val_loss'])   

y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)








                  



