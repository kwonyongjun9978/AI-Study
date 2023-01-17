from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
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

model=Sequential()
model.add(Dense(50, activation='relu', input_shape=(30,)))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])
'''
이진분류 : 마지막 y 값이 0, 1 중 하나만 나와야한다.
따라서 마지막 layer의 activation='sigmoid'(0~1)여야 한다.
그리고 loss='binary_crossentropy'이다. 또한, metrics=['accuracy'] 쓰기.
'''
from tensorflow.keras.callbacks import EarlyStopping 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=50, 
                              restore_best_weights=True, 
                              verbose=1)
model.fit(x_train, y_train, epochs=10000, batch_size=16,
          validation_split=0.2,
          callbacks=[earlyStopping],
          verbose=1)

#4. 평가, 예측
# loss=model.evaluate(x_test,y_test)
# print('loss, accuracy : ', loss) 
loss, accuracy = model.evaluate(x_test,y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)  

y_predict=model.predict(x_test)
'''
ValueError: Classification metrics can't handle a mix of binary and continuous targets
y_predict는 실수로 출력되어 있고 y_test에는 1,0으로만 되어있음.
따라서 자료형이 맞지 않다고 오류 뜨는 것임.
해결하는 방법은? y_predict 정수형으로 변환
'''
y_predict = y_predict.flatten() # 차원 펴주기
y_predict = np.where(y_predict > 0.5, 1 , 0) #0.5보다크면 1, 작으면 0

print(y_predict[:10]) #정수형으로 변환
print(y_test[:10])    

from sklearn.metrics import r2_score, accuracy_score
acc=accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

