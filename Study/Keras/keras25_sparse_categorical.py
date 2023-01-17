from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

#1. 데이터
datasets=load_iris()

x=datasets.data
y=datasets['target']
# print(x)
# print(y)
# print(x.shape, y.shape) #(150, 4) (150,)
# print(np.unique(y, return_counts=True)) #(array([0, 1, 2]), array([50, 50, 50], dtype=int64))

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
#sparse_categorical_crossentropy로 다중 분류할 때 softmax의 node의 개수는 one-hot 하지 않았지만 one-hot한 것처럼 y의 class 개수를 적어준다.
                                   
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
y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict, axis=1) 
print("y_predict(예측값) : ", y_predict)
#y_test=np.argmax(y_test, axis=1) #원핫을 안했으니까 여기서 쓸 필요가 없음
print("y_test(원래값) : ", y_test)
acc=accuracy_score(y_test, y_predict)
print(acc)



