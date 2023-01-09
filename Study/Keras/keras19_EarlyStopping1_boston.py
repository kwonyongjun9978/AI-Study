from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset=load_boston()
x=dataset.data
y=dataset.target

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)

# 2.모델구성
model=Sequential()
model.add(Dense(5, input_shape=(13,))) 
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

from tensorflow.keras.callbacks import EarlyStopping #대문자=class, 소문자=함수 
earlyStopping = EarlyStopping(monitor='val_loss', 
                              mode='min', 
                              patience=10, 
                              restore_best_weights=True, #restore_best_weights의 default값 : True
                              verbose=1)                                  
                                                                         
hist = model.fit(x_train, y_train, epochs=300, batch_size=1, 
                 validation_split=0.2, callbacks=[earlyStopping], 
                 verbose=1)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("===============================")
print(hist) #<keras.callbacks.History object at 0x00000231DF5D2AC0>
print("===============================")
print(hist.history) 
print("===============================")
print(hist.history['loss'])   
print("===============================")
print(hist.history['val_loss'])   

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) 
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid() 
plt.xlabel('epochs') 
plt.ylabel('loss')   
plt.title('boston loss') 
plt.legend()  
plt.show()




                  



