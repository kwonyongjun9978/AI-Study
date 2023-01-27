import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1.데이터
datasets=load_wine()
x=datasets.data
y=datasets.target

#print(datasets.DESCR)   

# print(x.shape, y.shape) #(178, 13) (178,)
# print(y)
# print(np.unique(y)) #[0 1 2] //y데이터에는 0,1,2값만 있다 -> 다중분류 확인  
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# 원핫인코딩
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,     
    #random_state=333
    stratify=y 
)

#Scaler(데이터 전처리) 
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(142, 13) (36, 13)

x_train = x_train.reshape(142,13,1)
x_test = x_test.reshape(36,13,1)

# 2.모델구성
model = Sequential()
model.add(LSTM(60, input_shape=(13,1), activation='relu')) 
model.add(Dense(50, activation='relu'))                                                                                                
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=40, 
                              restore_best_weights=True, 
                              verbose=2)

model.fit(x_train, y_train, epochs=1000, batch_size=2,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=2)

#4 평가, 예측
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

from sklearn.metrics import accuracy_score
import numpy as np
y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict)
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print("acc : ", acc)

'''
loss :  0.00304885720834136
accuracy :  1.0
y_predict(예측값) :  [1 0 0 2 2 0 1 1 1 0 1 2 2 0 0 1 2 1 2 0 1 0 2 0 2 1 2 0 2 1 1 1 1 1 0 0]
y_test(원래값) :  [1 0 0 2 2 0 1 1 1 0 1 2 2 0 0 1 2 1 2 0 1 0 2 0 2 1 2 0 2 1 1 1 1 1 0 0]
1.0
'''