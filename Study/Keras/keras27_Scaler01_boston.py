from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test) # fit.transform은 train데이터만 써야만한다

# 2.모델구성
model=Sequential()
model.add(Dense(256, activation='relu', input_shape=(13,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='relu'))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 

from tensorflow.keras.callbacks import EarlyStopping #대문자=class, 소문자=함수 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min',          
                              patience=40, 
                              restore_best_weights=True, 
                              verbose=1 )                                  
                                                                      
hist = model.fit(x_train, y_train, epochs=500, batch_size=2, 
                 validation_split=0.2, callbacks=[earlyStopping], 
                 verbose=1)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
r2=r2_score(y_test,y_predict)
print("R2 : ", r2)

'''
변환전
loss :  [23.091249465942383, 3.2509610652923584]
R2 :  0.7645647106714789

변환후(MinMaxScaler)
loss :  [15.005447387695312, 2.478259563446045]
R2 :  0.8470064684470626

변환후(StandardScaler)
loss :  [13.293885231018066, 2.3614625930786133]
R2 :  0.8644573420521556
'''




                  



