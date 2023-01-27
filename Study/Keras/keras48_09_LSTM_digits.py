import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=load_digits()

#print(datasets.DESCR)

x=datasets.data
y=datasets['target']
# print(x.shape, y.shape) #(1797, 64) (1797,)
# print(np.unique(y, return_counts=True))
'''
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))
'''

#이미지
# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[4])
# plt.show()

# 원핫인코딩
from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
# print(y)
# print(y.shape)

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

print(x_train.shape, x_test.shape) #(1437, 64) (360, 64)

x_train = x_train.reshape(1437,64,1)
x_test = x_test.reshape(360,64,1)

# 2.모델구성
model = Sequential()
model.add(LSTM(256, input_shape=(64,1), activation='relu'))                                                                                                
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=200, 
                              restore_best_weights=True, 
                              verbose=2)

model.fit(x_train, y_train, epochs=30000, batch_size=32,
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
print("y_predict(예측값) : ", y_predict)
y_test=np.argmax(y_test, axis=1)
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)

'''
dnn
loss :  0.3344237804412842
accuracy :  0.980555534362793

cnn
loss :  0.049522437155246735
accuracy :  0.9833333492279053
'''