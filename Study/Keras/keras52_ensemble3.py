#3개의 모델을 3개의 모델로 합치기
#1. 데이터
import numpy as np
x1_datasets = np.array([range(100), range(301,401)]).transpose()
print(x1_datasets.shape) #(100, 2) 
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
print(x2_datasets.shape) #(100, 3)  
x3_datasets = np.array([range(100,200), range(1301,1401)]).transpose()
print(x3_datasets.shape) #(100, 2)

y1 = np.array(range(2001, 2101)) #(100, )
y2 = np.array(range(201, 301)) #(100, )
y3 = np.array(range(501, 601)) #(100, )  
 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,\
    y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, y3, train_size=0.7, random_state=444
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape, y3_train.shape)   #(70, 2) (70, 3) (70, 2) (70,) (70,) (70,)               
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape, y3_test.shape)      #(30, 2) (30, 3) (30, 2) (30,) (30,) (30,)

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

#2-2. 모델2  
input2 = Input(shape=(3,))
dense21 = Dense(256, activation='linear', name='ds21')(input2)              
dense22 = Dense(128, activation='linear', name='ds22')(dense21)              
dense23 = Dense(64, activation='linear', name='ds23')(dense22)              
output2 = Dense(32, activation='linear', name='ds24')(dense23)              

#2-3. 모델3 
input3 = Input(shape=(2,))
dense1 = Dense(512, activation='relu', name='ds31')(input1)              
dense2 = Dense(256, activation='relu', name='ds32')(dense1)              
dense3 = Dense(128, activation='relu', name='ds33')(dense2)              
dense4 = Dense(64, activation='relu', name='ds34')(dense3)
output3 = Dense(32, activation='relu', name='ds35')(dense4)
          
#2-4. 모델병합
from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2,output3], name='mg1')
merge1 = Concatenate()([output1, output2, output3])
merge2 = Dense(128, activation='relu', name='mg2')(merge1)
merge3 = Dense(64, name='mg3')(merge2)
merge4 = Dense(32, name='mg4')(merge3)
last_output = Dense(1, name='last')(merge4)
model = Model(inputs=[input1, input2, input3], outputs=last_output)

model.summary()
'''
■ concatenate vs Concatenate
둘 다 같은 값으로 나오지만 concatenate를 사용할 때에는 단순히 concatenate([output1, output2])으로 쓰면 되고,
Concatenate를 사용할 때에는 Concatenate() 이렇게 () 괄호를 써줘야 한다.(name사용X)
'''
#2-5. 모델4 분기1.
dense5 = Dense(512, activation='relu', name='ds51')(last_output)              
dense5 = Dense(256, activation='relu', name='ds52')(dense5)              
dense5 = Dense(128, activation='relu', name='ds53')(dense5)              
dense5 = Dense(64, activation='relu', name='ds54')(dense5)
output5 = Dense(32, activation='relu', name='ds55')(dense5)

#2-6. 모델5 분기2.
dense6 = Dense(512, activation='relu', name='ds61')(last_output)              
dense6 = Dense(256, activation='relu', name='ds62')(dense6)              
dense6 = Dense(128, activation='relu', name='ds63')(dense6)              
dense6 = Dense(64, activation='relu', name='ds64')(dense6)
output6 = Dense(32, activation='relu', name='ds65')(dense6)

#2-7. 모델6 분기3.
dense7 = Dense(512, activation='relu', name='ds71')(last_output)              
dense7 = Dense(256, activation='relu', name='ds72')(dense7)              
dense7 = Dense(128, activation='relu', name='ds73')(dense7)              
dense7 = Dense(64, activation='relu', name='ds74')(dense7)
output7 = Dense(32, activation='relu', name='ds75')(dense7)

model = Model(inputs=[input1, input2, input3], outputs=[output5, output6, output7])

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
model.fit([x1_train, x2_train, x3_train], [y1_train,y2_train,y3_train], epochs=10, batch_size=1)

#4. 평가, 예측
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test,y2_test,y3_test])
print('loss : ', loss)

