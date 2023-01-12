from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=load_breast_cancer()
# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x=datasets['data']
y=datasets['target']
# print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
# model=Sequential()
# model.add(Dense(50, activation='linear', input_shape=(30,)))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# 2.모델구성(함수형)
input1=Input(shape=(30,))
dense1=Dense(50, activation='linear')(input1)
dense2=Dense(40, activation='relu')(dense1)
dense3=Dense(30, activation='relu')(dense2)
dense4=Dense(20, activation='relu')(dense3)
dense5=Dense(10, activation='relu')(dense4)
output1=Dense(1, activation='sigmoid')(dense5)
model=Model(inputs=input1, outputs=output1)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
#이진분류 : 마지막 activation은 sigmoid, loss는 binary_crossentropy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=50, 
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
                                  filepath=filepath+'k31_6_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping,ModelCheckpoint],
          verbose=1)

#4. 평가, 예측
# loss=model.evaluate(x_test,y_test)
# print('loss, accuracy : ', loss) 
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)  

y_predict=model.predict(x_test)

#y_predict 정수형으로 변환
y_predict = y_predict.flatten() # 차원 펴주기
y_predict = np.where(y_predict > 0.5, 1 , 0) #0.5보다크면 1, 작으면 0

print(y_predict[:10]) # -> 정수형으로 바꿔줘야한다
print(y_test[:10])    

from sklearn.metrics import r2_score, accuracy_score
acc=accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)


