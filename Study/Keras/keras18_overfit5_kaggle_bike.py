import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#1. 데이터
path='./_data/bike/' 
train_csv=pd.read_csv(path+'train.csv', index_col=0) 
test_csv=pd.read_csv(path+'test.csv', index_col=0) 
submission=pd.read_csv(path+'sampleSubmission.csv', index_col=0)

train_csv = train_csv.dropna()  #결측치 제거

x=train_csv.drop(['casual','registered','count'], axis=1)
#print(x) #[10886 rows x 8 columns]
y=train_csv['count']
#print(y)
#print(y.shape) #(10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    train_size=0.7,
    shuffle=True,
    random_state=44
)

print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)
print(y_train.shape, y_test.shape) #(7620,) (3266,)
print(submission.shape) 

#2.모델구성
model=Sequential()
model.add(Dense(32, input_dim=8, activation='relu')) 
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist=model.fit(x_train, y_train, epochs=100, batch_size=10,
               validation_split=0.2)

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("===============================")
print(hist) #
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
plt.title('bike loss') 
plt.legend() 
plt.show()