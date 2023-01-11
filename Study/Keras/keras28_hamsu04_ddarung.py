import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path='./_data/ddarung/' 
#2개 돌릴경우
# path='../_data/ddarung/' 

train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'submission.csv', index_col=0) 

#선형 방법을 이용하여 결측치
train_csv = train_csv.interpolate(method='linear', limit_direction='forward')

x=train_csv.drop(['count'], axis=1)
y=train_csv['count']

x_train, x_validation, y_train, y_validation = train_test_split(x, y,
    train_size=0.85, test_size=0.15, shuffle=False
)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
    test_size=0.3, shuffle=False
)
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
test_csv=scaler.transform(test_csv)

#2.모델구성
inputs = Input(shape=(9, ))
hidden1 = Dense(256, activation='relu') (inputs)
hidden2 = Dense(128, activation='relu') (hidden1)
hidden3 = Dense(64, activation='relu') (hidden2)
hidden4 = Dense(32) (hidden3)
hidden5 = Dense(16) (hidden4)
hidden6 = Dense(8) (hidden5)
output = Dense(1) (hidden6)

model = Model(inputs=inputs, outputs=output)

#3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping  
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=300, 
                              restore_best_weights=True, 
                              verbose=1)
hist=model.fit(x_train, y_train, epochs=10000, batch_size=8, validation_data=(x_validation, y_validation)
               ,callbacks=[earlyStopping], verbose=1)

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

submission.to_csv(path+'submission_011102.csv')



