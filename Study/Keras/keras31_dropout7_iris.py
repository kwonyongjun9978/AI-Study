from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model ,load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=load_iris()
#print(datasets.DESCR)         #클래스 개수 확인        
#print(datasets.feature_names)  

x=datasets.data
y=datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)

# 원핫인코딩
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y) 
# print(y.shape) # y=(150,) -> (150,3) 
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,      
    #random_state=333  
    stratify=y         
)
# print(y_train)
# print(y_test)
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(50, activation='relu', input_shape=(4,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(3, activation='softmax'))

# 2.모델구성(함수형)
input1=Input(shape=(4,))
dense1=Dense(50, activation='relu')(input1)
dense2=Dense(40, activation='relu')(dense1)
dense3=Dense(30, activation='relu')(dense2)
dense4=Dense(20, activation='relu')(dense3)
dense5=Dense(10, activation='relu')(dense4)
output1=Dense(3, activation='softmax')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary()
'''
다중분류
마지막 레이어의 activation은 무조건 softmax
클래스의 개수를 최종 Output layer 노드의 개수로 설정
loss 는 categorical_crossentropy
'''                                    
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=30, 
                              restore_best_weights=True, 
                              verbose=1)

import datetime
date = datetime.datetime.now() #현재 시간 반환
print(date) 
print(type(date)) #<class 'datetime.datetime'>
date=date.strftime("%m%d_%H%M") #문자열 타입으로 변환
print(date) 
print(type(date)) #<class 'str'>

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=1,
                                  save_best_only=True,
                                  filepath=filepath+'k31_7_'+date+'_'+filename)
model.fit(x_train, y_train, epochs=1000, batch_size=8,
          validation_split=0.2,
          callbacks=[earlyStopping,ModelCheckpoint],
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
print("   y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)


