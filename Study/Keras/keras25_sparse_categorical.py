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

# 원핫인코딩
# from tensorflow.keras.utils import to_categorical
# y=to_categorical(y)
# # print(y) 
# # print(y.shape) # y=(150,) -> (150,3) 
 
x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    shuffle=True,       
    #random_state=333  
    stratify=y         
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
#원핫은 하지 않았지만 loos에 sparse_categorical_crossentropy 사용하였으므로 자동으로 y의 컬럼을 맞춰준다
                                   
#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
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
#y_test=np.argmax(y_test, axis=1) #원핫을 안했으니까 여기서 쓸 필요가 없음
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)



