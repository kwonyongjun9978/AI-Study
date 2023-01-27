import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets['target']
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840,283301,35754,2747,9493,17367,20510],dtype=int64))

# 원핫인코딩
#방법3(scikit-onehotencoder)
# print(y.shape) #(581012,)
# print(type(y)) #<class 'numpy.ndarray'>
y = y.reshape(581012, 1)
# print(y.shape) #(581012, 1)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y)
# print(y[:15])
# print(type(y)) #<class 'scipy.sparse._csr.csr_matrix'>
# print(y.shape) #(581012, 7)
y=y.toarray()
# print(y[:15])
# print(type(y)) #<class 'numpy.ndarray'>
# print(y.shape) #(581012, 7)

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

print(x_train.shape, x_test.shape) #(464809, 54) (116203, 54)

x_train = x_train.reshape(464809,54,1)
x_test = x_test.reshape(116203,54,1)

# 2.모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(54,1), activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, 
                              restore_best_weights=True, 
                              verbose=2)

model.fit(x_train, y_train, epochs=500, batch_size=3000,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=2)


#4 평가, 예측
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

from sklearn.metrics import accuracy_score

y_predict=model.predict(x_test)

y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict[:20])
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test[:20])
acc=accuracy_score(y_test, y_predict)
print("acc(정확도) : ", acc)

'''
loss :  0.24077723920345306
accuracy :  0.9054241180419922
acc(정확도) :  0.9054241284648417
'''
