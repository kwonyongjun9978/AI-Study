#1개의 모델을 2개의 모델로 합치기
#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100, 2) 

y1 = np.array(range(2001, 2101)) #(100, )
y2 = np.array(range(201, 301)) #(100, ) 
 
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datasets, y1, y2, train_size=0.7, random_state=444
)

print(x1_train.shape, y1_train.shape, y2_train.shape) #(70, 2) (70,) (70,)              
print(x1_test.shape, y1_test.shape, y2_test.shape) #(30, 2) (30,) (30,)     

#2. 모델구성 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델1
input1 = Input(shape=(2,))
dense1 = Dense(512, activation='relu', name='ds11')(input1)              
dense2 = Dense(256, activation='relu', name='ds12')(dense1)              
dense3 = Dense(128, activation='relu', name='ds13')(dense2)              
dense4 = Dense(64, activation='relu', name='ds14')(dense3)
output1 = Dense(32, activation='relu', name='ds15')(dense4)
       
#2-2. 모델병합
from tensorflow.keras.layers import concatenate 
merge1 = concatenate([output1], name='mg1')
merge2 = Dense(128, activation='relu', name='mg2')(merge1)
merge3 = Dense(64, name='mg3')(merge2)
merge4 = Dense(32, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)
model = Model(inputs=input1, outputs=last_output)

model.summary()

#2-3. 모델2 분기1.
dense5 = Dense(512, activation='relu', name='ds51')(last_output)              
dense5 = Dense(256, activation='relu', name='ds52')(dense5)              
dense5 = Dense(128, activation='relu', name='ds53')(dense5)              
dense5 = Dense(64, activation='relu', name='ds54')(dense5)
output5 = Dense(32, activation='relu', name='ds55')(dense5)

#2-4. 모델3 분기2.
dense6 = Dense(512, activation='relu', name='ds61')(last_output)              
dense6 = Dense(256, activation='relu', name='ds62')(dense6)              
dense6 = Dense(128, activation='relu', name='ds63')(dense6)              
dense6 = Dense(64, activation='relu', name='ds64')(dense6)
output6 = Dense(32, activation='relu', name='ds65')(dense6)

model = Model(inputs=input1, outputs=[output5, output6])

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train,y2_train], epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x1_test, [y1_test,y2_test])
print('loss : ', loss)
