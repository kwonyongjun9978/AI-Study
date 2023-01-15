import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
'''
x, y로 훈련했는데 평가 때에도 x, y로 평가함. 즉, 같은 값으로 평가함.
해결 방법: 평가할 때에는 훈련했을 때에와 다른 값으로 평가해야함.
          데이터를 쪼개서 70~80%의 데이터로 훈련시키고 나머지 20%의 데이터로 평가시킨다.
          데이터를 받으면 train set, validation set, test set으로 나누기.
          
가장 적절한 모델을 찾기 위해 데이터를 train data와 test data로 나눈 뒤,
trian data에 각각의 모델로 학습시켜 test data로 각 모델의 최종 정확도를 확인 

data를 train과 test로 나눠서 train으로만 학습하면서 최적의 매개변수(가중치)를 찾고, 
그런 다음 test를 사용하여 모델의 실력을 평가         
'''
#1.데이터
# x=np.array([1,2,3,4,5,6,7,8,9,10]) //(10, ) cf)([[1,2,3,4,5,6,7,8,9,10]]) (10,1)
# y=np.array(range(10))              //(10, )
#y=wx+b //w=1,b=-1

'''
전체데이터를 훈련시키면 과적합(overfit)발생 -> 훈련데이터(train set)와 평가데이터(test set)로 분리해서 관리 
|-------|---|
'''
X_train=np.array([1,2,3,4,5,6,7]) #(7,) 특성(열)이 하나, (7,1)과는 같지는 않음
X_test=np.array([8,9,10])         #(3,)
Y_train=np.array(range(7))
Y_test=np.array(range(7,10))

#2.모델구성
model=Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(9))
model.add(Dense(13))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(X_train,Y_train,epochs=200, batch_size=1) #훈련데이터 대입

#4.평가, 예측
loss=model.evaluate(X_test,Y_test) #평가데이터는 실질적으로 훈련에 관여하지 않는다.
print('loss : ', loss)
result=model.predict([11])
print('[11]의 결과', result)

'''
Epoch 200/200
7/7 [==============================] - 0s 709us/step - loss: 0.0607  //훈련데이터의 loss값
1/1 [==============================] - 0s 81ms/step - loss: 0.0733   //평가데이터의 loss값
loss :  0.07328144460916519 //평가데이터의 loss값
[11]의 결과 [[10.089934]]
'''