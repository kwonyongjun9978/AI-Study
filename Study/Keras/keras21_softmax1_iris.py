from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets=load_iris()
# print(datasets.DESCR)          #pandas .describe()  / .info()
# print(datasets.feature_names)  #pandas .columns

x=datasets.data
y=datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)

# 원핫인코딩(범주형 데이터를 1과 0의 데이터로 바꿔주는 과정)
# y=(150,) -> (150,3) = y값(의 클래스)의 개수 만큼 컬럼이 늘어난다
# 방법1
# import pandas as pd
# y = pd.get_dummies(y)
# print(y)
# print(y.shape)
# 방법2
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y)
# print(y.shape)
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,   #False의 문제점 : 데이터가 한쪽으로 몰려있을 경우   
    #random_state=333
    stratify=y #데이터를 일정하게 배분(분류형 데이터에서만 사용 가능)
)
# print(y_train)
# print(y_test)

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
y_predict=np.argmax(y_predict, axis=1) #argmax : 가장 큰 값을 찾아내는 함수
print("y_predict(예측값) : ", y_predict)
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)
