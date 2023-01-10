from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset=load_boston()
x=dataset.data
y=dataset.target

# print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test=train_test_split(
    x,y,
    train_size=0.8,
    shuffle=True,
    random_state=333
)

# 2.모델구성
model=Sequential()
#model.add(Dense(5, input_dim=13))     #input_dim : 2차원(행렬)
model.add(Dense(5, input_shape=(13,))) #input_shape : 다차원 ex(100,10,5) -> (10,5)
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3.컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam') 
start=time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, 
          verbose=0)  
end=time.time()
#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("걸린시간 : ", end-start)

# verbose=1 걸린시간 : 17.64365267753601  (그대로, 기본값(default))
# verbose=0 걸린시간 : 15.846755266189575 (아무것도 안나옴, 1과 비교했을때 시간 단축)
# verbose=2 걸린시간 : 16.91422152519226  (progress bar 제거)
# verbose=3 걸린시간 : 16.591039180755615 (Epoch만 나옴)
# verbose=4 걸린시간 : 16.742960929870605 (4이상부터는 3과 동일하게 Epoch만 나옴)

