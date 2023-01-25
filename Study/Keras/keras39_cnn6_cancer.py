from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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

#Scaler(데이터 전처리) 
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(455, 30) (114, 30)

x_train = x_train.reshape(455,30,1,1)
x_test = x_test.reshape(114,30,1,1)

# 2.모델구성
model=Sequential()
model.add(Conv2D(50, (6,1), input_shape=(30,1,1), padding='same',
                 activation='relu'))
model.add(Conv2D(filters=40, kernel_size=(4,1), padding='same'))  
model.add(Conv2D(filters=30, kernel_size=(2,1))) 
model.add(Flatten())
model.add(Dense(30, activation='relu')) 
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))                                              
model.add(Dense(1, activation='sigmoid')) 
model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=50, 
                              restore_best_weights=True, 
                              verbose=2)

import datetime
date = datetime.datetime.now() 
print(date) 
print(type(date)) 
date=date.strftime("%m%d_%H%M") 
print(date) 
print(type(date)) 

filepath='./_save/MCP/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5' 

#ModelCheckpoint
ModelCheckpoint = ModelCheckpoint(monitor='val_loss',
                                  mode='auto',
                                  verbose=2,
                                  save_best_only=True,
                                  filepath=filepath+'k39_6_'+date+'_'+filename)

model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping,ModelCheckpoint],
          verbose=2)

#4. 평가, 예측
# loss=model.evaluate(x_test,y_test)
# print('loss, accuracy : ', loss) 
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)  

y_predict=model.predict(x_test)

#y_predict 정수형으로 변환
y_predict = y_predict.flatten() 
y_predict = np.where(y_predict > 0.5, 1 , 0) 

print(y_predict[:10]) 
print(y_test[:10])    

from sklearn.metrics import r2_score, accuracy_score
acc=accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

'''
dnn
loss :  0.13065771758556366
accuracy :  0.9736841917037964
accuracy_score :  0.9736842105263158

cnn
loss :  0.10732833296060562
accuracy :  0.9736841917037964
accuracy_score :  0.9736842105263158
'''