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

#Scaler(데이터 전처리) 
scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

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
                                  filepath=filepath+'k31_6_'+date+'_'+filename)

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
Epoch 00063: val_loss did not improve from 0.04396
4/4 [==============================] - 0s 893us/step - loss: 0.1307 - accuracy: 0.9737
loss :  0.13065771758556366
accuracy :  0.9736841917037964
accuracy_score :  0.9736842105263158
'''
