from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten ,Dropout
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
    random_state=444
)

#Scaler(데이터 전처리) 
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(353, 10) (89, 10)

x_train = x_train.reshape(353,10,1)
x_test = x_test.reshape(89,10,1)

# 2.모델구성
model = Sequential()
model.add(Conv1D(256, 2, input_shape=(10,1), activation='relu'))                                                                                                
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
                              patience=250, 
                              restore_best_weights=True, 
                              verbose=2)

hist = model.fit(x_train, y_train, epochs=5000, batch_size=8, 
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
dnn
loss :  3146.67626953125
RMSE :  56.09524152540849
R2 :  0.4962242406396765

cnn
loss :  2894.641845703125
RMSE :  53.80187594508866
R2 :  0.5365743590244026

LSTM
loss :  3218.5263671875
RMSE :  56.73205759983632
R2 :  0.48472117942040394

Conv1D
loss :  2920.545166015625
RMSE :  54.04206902364643
R2 :  0.5324272879702632
'''






                  



