import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets['target']
print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840,283301,35754,2747,9493,17367,20510],dtype=int64))

# 원핫인코딩
#방법1
# from tensorflow.keras.utils import to_categorical
# y=to_categorical(y)
# print(type(y))
# 힌트 np.delete

#방법2
# import pandas as pd
# y=pd.get_dummies(y)
# print(type(y))
#힌트 .values .numpy()

#방법3
from sklearn.preprocessing import OneHotEncoder
# print(y)
# print(y.shape)
y=y.reshape(-1,1)
#print(y)
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
#print(y)
y=y.toarray()
# print(y)
# print(y.shape) #(581012, 7)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,     
    #random_state=333
    stratify=y 
)

#2. 모델구성
model=Sequential()
model.add(Dense(256, activation='relu', input_shape=(54,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
import time 
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=500, 
                              restore_best_weights=True, 
                              verbose=1)
start=time.time() 
model.fit(x_train, y_train, epochs=300000, batch_size=500,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)
end=time.time()
#4 평가, 예측
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

from sklearn.metrics import accuracy_score

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict)
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)

print("걸린시간 : ", end-start)
