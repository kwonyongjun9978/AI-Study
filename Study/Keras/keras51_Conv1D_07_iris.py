from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model ,load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten ,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets=load_iris()
#print(datasets.DESCR)              
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

#Scaler(데이터 전처리) 
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

print(x_train.shape, x_test.shape) #(120, 4) (30, 4)

x_train = x_train.reshape(120,4,1)
x_test = x_test.reshape(30,4,1)

# 2.모델구성
model = Sequential()
model.add(Conv1D(60, 2, input_shape=(4,1), activation='relu')) 
model.add(Dense(50, activation='relu'))                                                                                                
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
                                  
#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=30, 
                              restore_best_weights=True, 
                              verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=8,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=2)

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
print("acc : ", acc)

'''
loss :  0.01556424330919981
accuracy :  1.0
y_predict(예측값) :  [1 2 1 2 2 1 0 0 0 0 2 1 0 1 2 1 0 2 0 2 0 2 2 1 1 1 0 1 0 2]
   y_test(원래값) :  [1 2 1 2 2 1 0 0 0 0 2 1 0 1 2 1 0 2 0 2 0 2 2 1 1 1 0 1 0 2]
acc : 1.0
'''


