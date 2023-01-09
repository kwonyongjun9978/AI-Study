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
                              restore_best_weights=True, 
                              verbose=1 )                                  
'''
1.monitor : EarlyStopping의 기준이 되는 값을 입력합니다.
2.min_delta : 개선된 것으로 간주하기 위한 최소한의 변화량입니다.
              예를 들어, min_delta가 0.01이고, 30에폭에 정확도가 0.8532라고 할 때,
              만약 31에폭에 정확도가 0.8537라고 하면 이는 0.005의 개선이 있었지만 min_delta 값 0.01에는 미치지 못했으므로 개선된 것으로 보지 않습니다.
3.patience : Training이 진행됨에도 더이상 monitor되는 값의 개선이 없을 경우, 최적의 monitor 값을 기준으로 몇 번의 epoch을 진행할 지 정하는 값. 
             예를 들어 patience는 3이고, 30에폭에 정확도가 99%였을 때,
             만약 31번째에 정확도 98%, 32번째에 98.5%, 33번째에 98%라면 더 이상 Training을 진행하지 않고 종료시킵니다.     
4.verbose : 0 또는 1
            1일 경우, EarlyStopping이 적용될 때, 화면에 적용되었다고 나타냅니다.
            0일 경우, 화면에 나타냄 없이 종료합니다.
5.mode : "auto" 또는 "min" 또는 "max"
          monitor되는 값이 최소가 되어야 하는지, 최대가 되어야 하는지 알려주는 인자입니다.
          예를 들어, monitor하는 값이 val_acc 즉 정확도일 경우, 값이 클수록 좋기때문에 "max"를 입력하고, val_loss일 경우 작을수록 좋기 때문에 "min"을 입력합니다.
         "auto"(기본값)는 모델이 알아서 판단합니다.
6.baseline : 모델이 달성해야하는 최소한의 기준값을 선정합니다.
             patience 이내에 모델이 baseline보다 개선됨이 보이지 않으면 Training을 중단시킵니다.
             예를 들어 patience가 3이고 baseline이 정확도기준 0.98 이라면,
             3번의 trianing안에 0.98의 정확도를 달성하지 못하면 Training이 종료됩니다.
7.restore_best_weights : True(기본값), False
                         True라면 training이 끝난 후, model의 weight를 monitor하고 있던 값이 가장 좋았을 때의 weight로 복원합니다.
                         False라면, 마지막 training이 끝난 후의 weight로 놔둡니다.                                                        
'''                                                                         
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




                  



