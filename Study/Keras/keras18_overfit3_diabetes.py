from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset=load_diabetes()
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
model.add(Dense(88, input_dim=10, activation='relu'))
model.add(Dense(168, activation='relu'))
model.add(Dense(208, activation='relu'))
model.add(Dense(248, activation='relu'))
model.add(Dense(288, activation='relu'))
model.add(Dense(240, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 
hist = model.fit(x_train, y_train, epochs=600, batch_size=25, validation_split=0.2, verbose=1)  

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)
'''
print("===============================")
print(hist) #
print("===============================")
print(hist.history) 
print("===============================")
print(hist.history['loss'])   
print("===============================")
print(hist.history['val_loss'])   
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) 
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid() 
plt.xlabel('epochs') 
plt.ylabel('loss')   
plt.title('diabetes loss') 
plt.legend() 
plt.show()




                  



