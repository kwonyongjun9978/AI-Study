import numpy as np 
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv1D,Input,\
    Dropout,LSTM, Conv2D, Flatten, concatenate
from sklearn.model_selection import train_test_split

x_datasets = np.array([range(100), range(301,401)]).transpose()
y1 = np.array(range(2001,2101)) # 삼성
y2 = np.array(range(201,301))  # 아모레
y3 = np.array(range(201,301))  
x1_train, x1_test,\
    y1_train, y1_test,y2_train,y2_test,y3_train,y3_test,\
        =train_test_split(x_datasets,
        y1,y2,y3,train_size=0.7,
        random_state=1234)

# Model 1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
dense4 = Dense(14, activation='relu', name='ds14')(dense3)

# Concatenate the outputs of the two models
from tensorflow.keras.layers import Concatenate,concatenate
# merge = Concatenate()([dense4 ])
merge = concatenate([dense4 ])
merge2 = Dense(12, activation='relu', name='mg2')(merge)
merge3 = Dense(13, activation='relu', name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)
# Model 2
dense51 = Dense(21, activation='relu', name='ds51')(last_output)
dense52 = Dense(22, activation='relu', name='ds52')(dense51)
output5 = Dense(1, activation='relu', name='ds53')(dense52)

# Model 3
dense61 = Dense(21, activation='relu', name='ds61')(last_output)
dense62 = Dense(22, activation='relu', name='ds62')(dense61)
output6 = Dense(1, activation='relu', name='ds63')(dense62)


dense71 = Dense(21, activation='relu', name='ds71')(last_output)
dense72 = Dense(22, activation='relu', name='ds72')(dense71)
output7 = Dense(1, activation='relu', name='ds73')(dense72)


model = Model(inputs=input1, outputs=[output5,output6,output7])
model.compile(loss = 'mse', optimizer='adam',metrics='mae')
model.fit(x1_train, [y1_train,y2_train,y3_train], epochs=10, batch_size=32)

loss = model.evaluate(x1_test, [y1_test,y2_test,y3_test])
print('Loss:', loss)

y_pred = model.predict(x1_test)
print('Prediction:', y_pred)