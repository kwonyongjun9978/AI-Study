import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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

scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(256, activation='relu', input_shape=(13,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# 2.모델구성(함수형)
input1=Input(shape=(13,))
dense1=Dense(256, activation='relu')(input1)
dense2=Dense(128, activation='relu')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
output1=Dense(3, activation='softmax')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=40, 
                              restore_best_weights=True, 
                              verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=2,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

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
print(acc)
