from  sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#1.데이터
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=123
)
#2.모델구성
inputs=Input(shape=(10, ))
hidden1= Dense(256, activation='relu') (inputs)
hidden2=Dense(128, activation='relu') (hidden1)
hidden3=Dense(85, activation='relu') (hidden2)
hidden4=Dense(64, activation='relu') (hidden3)
hidden5=Dense(10, activation='relu') (hidden4)
hidden6=Dense(5, activation='relu') (hidden5)
output=Dense(1) (hidden6)

Model= Model(inputs=inputs,outputs=output)


#3.컴파일, 훈련
Model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
Model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.25)

#4.평가,예측
loss=Model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict=Model.predict(x_test)

def RMSE(y_test, y_predict):  #RMSE라는 함수를 정의
    return np.sqrt(mean_squared_error(y_test, y_predict))    
print("RMSE : ", RMSE(y_test, y_predict))            

r2=r2_score(y_test,y_predict)
print("R2 : ", r2)


