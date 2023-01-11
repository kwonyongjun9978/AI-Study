import numpy as np
import tensorflow as tf
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
#<1.keras-to_categorical vs 2.pandas-get_dummies vs 3.scikit-onehotencoder>
#방법1(keras-to_categorical)
# from tensorflow.keras.utils import to_categorical
# y=to_categorical(y)
# #print(y.shape) #(581012, 8)
# #print(type(y)) #<class 'numpy.ndarray'>
# #print(y[:10])
# #print(np.unique(y[:, 0], return_counts=True)) #모든 행의 0번째 컬럼 #(array([0.], dtype=float32), array([581012], dtype=int64))
# #print(np.unique(y[:, 1], return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))
# y=np.delete(y,0,axis=1)
# #print(y.shape) #(581012, 7)
# #print(y[:10])
# #print(np.unique(y[:, 0], return_counts=True)) #(array([0., 1.], dtype=float32), array([369172, 211840], dtype=int64))

'''
방법1
to_categorical의 특성 : 무조건 0부터 시작하게끔 한다. => 0이 없을 경우 class 하나 더 만듦.
y 데이터가 [1 2 3 4 5 6 7]일 경우 to_categorical(y)하면 [0 1 2 3 4 5 6 7]로 0을 더 추가해 만듦.
=> 해결방법: 첫번째 칼럼 삭제하기!
=> y = np.delete(y, 0, axis=1)
np.delete(데이터, 0번째, 행삭제는 axis=0, 열삭제는 axis=1)
'''

#방법2(pandas-get_dummies)
# import pandas as pd
# y=pd.get_dummies(y)
# print(y[:10])
# #(방법2-2)
# #print(type(y)) #<class 'pandas.core.frame.DataFrame'>(판다스에서는 헤더와 인덱스가 자동생성된다)
# y=y.values
# #print(type(y)) #<class 'numpy.ndarray'>
# #print(y.shape)
# #y=y.to_numpy() 로 해줘도 된다.
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


'''
방법3
OneHotEncoder : 명목변수든 순위변수든 모두 원핫인코딩을 해준다.
=> 해결방법: shape 맞추기
1) 스칼라: 원본 데이터를 y, y.shape, type(y)를 print 해보면
(581012,) 스칼라 형태의 numpy.ndarray 임을 알 수 있다.
2) 벡터: 원핫엔코더하려면 벡터 형태로 reshape 해줘야 한다.
y = y.reshape(581012,1) 해서 (581012, 1) 벡터 형태를 만든다.
# (-1,1) 하면 (전체, 1)과 같다.
3) scikit-learn에서 OneHotEncoder 가져오기
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
4) 원핫엔코딩: y = ohe.fit_transform(y)로 원핫엔코딩한다.
y = ohe.fit_transform(y) 하면 (581012, 7) 벡터 형태의 scipy.sparse._csr.csr_matrix가 나온다.
5) 데이터형태 바꾸기 : scipy CSR matrix 를 Numpy ndarray로 바꾼다.
y = y.toarray() 하면 데이터 종류만 numpy ndarray로 바뀐다.
'''

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
                              patience=15, 
                              restore_best_weights=True, 
                              verbose=1)
start=time.time() 
model.fit(x_train, y_train, epochs=100, batch_size=2000,
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
