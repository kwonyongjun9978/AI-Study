from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

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

# 2.모델구성
model=Sequential()
model.add(Dense(88, input_dim=10, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(208, activation='relu'))
model.add(Dense(248, activation='relu'))
model.add(Dense(288, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=50, 
                              restore_best_weights=True, 
                              verbose=1)

hist = model.fit(x_train, y_train, epochs=600, batch_size=25, 
                 validation_split=0.25, callbacks=[earlyStopping], 
                 verbose=1)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("===============================")
print(hist) 
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

'''
RMSE :  55.361665725767836
R2 :  0.4219220937064717
'''






                  



