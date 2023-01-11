from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
model=Sequential()
model.add(Dense(50, activation='relu', input_shape=(4,)))
model.add(Dense(40, activation='relu'))
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

'''
MinMaxScaler
loss :  0.05111243203282356
accuracy :  0.9666666388511658
'''
'''
StandardScaler
loss :  0.09314523637294769
accuracy :  0.9666666388511658
'''
