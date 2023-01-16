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
hist = model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=2)  
#model.fit은 훈련의 결과값을 반환하고 그걸 hist(history)라 하자.

#4.평가,예측
loss=model.evaluate(x_test, y_test) 
print('loss : ', loss)

print("===============================")
print(hist) #<keras.callbacks.History object at 0x00000231DF5D2AC0>
print("===============================")
print(hist.history) #model.fit에서 반환하는 loss와 val-loss의 변화량
                    #dictionary(key,value값으로 이루어짐)형태의 자료형
print("===============================")
print(hist.history['loss'])   
print("===============================")
print(hist.history['val_loss'])   

import matplotlib.pyplot as plt
#title 한글로 변경
import matplotlib as mpl
plt.rc('font', family='Malgun Gothic')     
mpl.rcParams['axes.unicode_minus'] = False 

plt.figure(figsize=(9,6)) #판사이즈(figsize(가로길이, 세로길이) 단위는 inch이다.)
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid() #격자
plt.xlabel('epochs') #x축 라벨
plt.ylabel('loss')   #y축 라벨
plt.title('보스턴 손실함수') #제목 이름 표시
plt.legend() #선의 이름
#plt.legend(loc='upper left') 
plt.show()

'''
<loss, val_loss를 통해 훈련이 잘 되는지 확인하기>
loss값을 참고하되 val_loss가 기준이 된다.
val_loss가 들쭉날쭉하면 훈련이 잘 안되는 것이다.
val_loss가 최소인 지점이 최적의 weight 점이다.
'''




                  



