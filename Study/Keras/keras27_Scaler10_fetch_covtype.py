import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=fetch_covtype()
x=datasets.data
y=datasets['target']
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts=True)) #(array([1, 2, 3, 4, 5, 6, 7]), array([211840,283301,35754,2747,9493,17367,20510],dtype=int64))

# 원핫인코딩
#방법2(pandas-get_dummies)
import pandas as pd
y=pd.get_dummies(y)
print(y[:10])
#(방법2-2)
#print(type(y)) #<class 'pandas.core.frame.DataFrame'>(판다스에서는 헤더와 인덱스가 자동생성된다)
y=y.values
#print(type(y)) #<class 'numpy.ndarray'>
#print(y.shape)
#y=y.to_numpy() 로 해줘도 된다.

# #y = np.array(y) #(방법2-1)

'''
방법2
(방법2-2)
get_dummies : 명목변수만 원핫인코딩을 해준다.
=> 해결방법: 자료형 확인
=> print(type()) 으로 자료형을 확인
y_predict는 <class 'numpy.ndarray'>
y_test는 <class 'pandas.core.frame.DataFrame'>가 나온다.
즉, y_test의 Dataframe을 numpy.ndarray로 바꿔줘야한다.
=> .values 로 pandas DataFrame을 Numpy ndarray로 바꿔주거나
=> .to_numpy() 로 pandas DataFrame을 Numpy ndarray로 바꿔주기.

(방법2-3)
get_dummies를 쓰면 자료형이 <class 'pandas.core.frame.DataFrame'>이다.
여기서 굳이 자료형을 <class 'numpy.ndarray'>로 바꾸지 않고
np.argmax를 tf.argmax로 바꿔서 결과를 구할수도 있다.
대신 마지막 결과에 나오는 데이터형이 <class 'numpy.ndarray'>가 아니라
<class 'tensorflow.python.framework.ops.EagerTensor'> 이다.
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

'''
MinMaxScaler
loss :  0.23466132581233978
accuracy :  0.9079025387763977
y_predict(예측값) :  [1 0 1 2 1 1 0 4 1 0 1 1 0 6 1 0 1 1 2 1]
y_test(원래값) :  [1 0 1 2 0 1 0 4 1 0 1 1 0 6 1 0 0 0 2 1]
acc(정확도) :  0.9079025498481106
걸린시간 :  428.7227509021759
'''

'''
StandardScaler
loss :  0.15783922374248505
accuracy :  0.9406125545501709
y_predict(예측값) :  [1 0 0 0 1 5 1 0 0 1 2 6 1 0 2 1 1 1 1 1]
y_test(원래값) :  [1 0 0 1 0 5 0 0 0 1 2 6 1 0 2 1 1 1 1 1]
acc(정확도) :  0.9406125487293787
걸린시간 :  207.72581696510315
'''

