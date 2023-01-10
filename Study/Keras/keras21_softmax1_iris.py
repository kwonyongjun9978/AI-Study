from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets=load_iris()
#print(datasets.DESCR)         #클래스 개수 확인        
#print(datasets.feature_names)  

x=datasets.data
y=datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)

# 원핫인코딩(범주형 데이터를 1과 0의 데이터로 바꿔주는 과정)
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y) 
# print(y.shape) # y=(150,) -> (150,3) = y값(의 클래스)의 개수 만큼 컬럼이 늘어난다
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,      #False 의 문제점은 데이터 분할 시, 한쪽으로 쏠림 현상 발생으로 데이터의 훈련도의 오차가 심해진다. 
    #random_state=333  #random_state 를 사용 시, 분리된 데이터가 비율이 안맞는 현상 발생
    stratify=y         #분리된 데이터가 비율이 일정하게 됨, 데이터 자체(y)가 분류형 일 때만 가능 , load_boston 데이터는 회귀형 데이터라 안됨.
)
print(y_train)
print(y_test)

#2. 모델구성
model=Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
'''
다중분류
마지막 레이어의 activation은 무조건 softmax
클래스의 개수를 최종 Output layer 노드의 개수로 설정
loss 는 categorical_crossentropy
'''                                    
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=25, 
                              restore_best_weights=True, 
                              verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=10,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4 평가, 예측
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy) 

# print(y_test[:5])
# y_predict=model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict)
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)

'''
*파이썬 넘파이 argmax, argmin 함수*
= 파이썬 넘파이 라이브러리에서 제공하는 최대값, 최소값의 위치 인덱스를 반환하는 함수
np.argmax : 함수 내에 array와 비슷한 형태(리스트 등 포함)의 input을 넣어주면 가장 큰 원소의 인덱스를 반환하는 형식입니다.
            다만, 가장 큰 원소가 여러개 있는 경우 가장 앞의 인덱스를 반환
np.argmin : np.argmax 와 반대로 최소값의 인덱스를 반환하는 함수
<==================================================================================================================>
*axis*
axis=1 행, axis=0 열 기준으로 계산한다는 의미           
'''