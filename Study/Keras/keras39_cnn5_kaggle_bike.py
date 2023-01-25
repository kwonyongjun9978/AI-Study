import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path='./_data/bike/' 
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

train_csv = train_csv.dropna()

x=train_csv.drop(['casual','registered','count'], axis=1)
y=train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
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

test_csv=scaler.transform(test_csv)

print(x_train.shape, x_test.shape) #(8708, 8) (2178, 8)
print(test_csv.shape) #(6493, 8)

x_train = x_train.reshape(8708,8,1,1)
x_test = x_test.reshape(2178,8,1,1)
test_csv = test_csv.reshape (6493,8,1,1)

# 2.모델구성
model=Sequential()
model.add(Conv2D(512, (4,1), input_shape=(8,1,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(2,1), padding='same'))  
model.add(Conv2D(filters=128, kernel_size=(2,1))) 
model.add(Flatten())
model.add(Dense(128, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))                                              
model.add(Dense(1, activation='relu')) 
model.summary()

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=300, 
                              restore_best_weights=True, 
                              verbose=2)  

hist = model.fit(x_train, y_train, epochs=10000, batch_size=64,
                 validation_split=0.2, callbacks=[earlyStopping], 
                 verbose=2)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
y_predict=model.predict(x_test)

def RMSE(y_test, y_predict):  
    return np.sqrt(mean_squared_error(y_test, y_predict))              
print("RMSE : ", RMSE(y_test, y_predict)) 

#제출
y_submit=model.predict(test_csv)

submission['count']=y_submit

submission.to_csv(path+'submission_012501.csv')

'''
dnn
loss :  20308.974609375
RMSE :  142.50955986002919

cnn
loss :  20443.77734375
RMSE :  142.98174486630552
'''



