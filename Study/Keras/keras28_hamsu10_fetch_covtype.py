import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets['target']
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840,283301,35754,2747,9493,17367,20510],dtype=int64))

# 원핫인코딩
#<1.keras-to_categorical vs 2.pandas-get_dummies vs 3.scikit-onehotencoder>
#방법1(keras-to_categorical)
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
#print(y.shape) #(581012, 8)
#print(type(y)) #<class 'numpy.ndarray'>
# print(y[:10])
# print(np.unique(y[:, 0], return_counts=True)) #모든 행의 0번째 컬럼 #(array([0.], dtype=float32), array([581012], dtype=int64))
# print(np.unique(y[:, 1], return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
y=np.delete(y,0,axis=1)
# print(y.shape) #(581012, 7)
# print(y[:10])
# print(np.unique(y[:, 0], return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))

'''
방법1
to_categorical의 특성 : 무조건 0부터 시작하게끔 한다. => 0이 없을 경우 class 하나 더 만듦.
y 데이터가 [1 2 3 4 5 6 7]일 경우 to_categorical(y)하면 [0 1 2 3 4 5 6 7]로 0을 더 추가해 만듦.
=> 해결방법: 첫번째 칼럼 삭제하기!
=> y = np.delete(y, 0, axis=1)
np.delete(데이터, 0번째, 행삭제는 axis=0, 열삭제는 axis=1)
'''
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,     
    #random_state=333
    stratify=y 
)

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(256, activation='relu', input_shape=(54,)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# 2.모델구성(함수형)
input1=Input(shape=(54,))
dense1=Dense(256, activation='relu')(input1)
dense2=Dense(128, activation='relu')(dense1)
dense3=Dense(64, activation='relu')(dense2)
dense4=Dense(32, activation='relu')(dense3)
dense5=Dense(16, activation='relu')(dense4)
output1=Dense(7, activation='softmax')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
import time 
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=20, 
                              restore_best_weights=True, 
                              verbose=1)
start=time.time() 
model.fit(x_train, y_train, epochs=500, batch_size=3000,
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
print("y_predict(예측값) : ", y_predict[:20])
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test[:20])
acc=accuracy_score(y_test, y_predict)
print("acc(정확도) : ", acc)

print("걸린시간 : ", end-start)


