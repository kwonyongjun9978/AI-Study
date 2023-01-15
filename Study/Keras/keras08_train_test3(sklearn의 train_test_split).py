import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1.데이터
x=np.array([1,2,3,4,5,6,7,8,9,10]) 
y=np.array(range(10))  
            
'''
데이터 분할할 때 앞에 train, 뒤에 test로 딱 나눠서 자르면 train에서 생긴 오차가 test에 그대로 영향을 준다. 이는 데이터 양이 많아질수록 더 심해진다.
예를 들어, 여기서 weight가 0.999라고 해도 데이터가 1000만개 넘어가면 뒤에 오차가 점점 벌어짐. 
따라서 데이터 분할할 때 전체 데이터 범위내에서 train과 test를 분리함 (즉, 한 칸씩 띄어서 잡아주거나 여러칸씩 띄어서 잡아줌)
데이터 전체 내에서 train과 test를 분리하면 과적합 문제가 생길 수 있다.
'''
#사이킷런 사용
'''
|----------|
    |----------|
'''
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test =train_test_split(   # ()=함수
    x,y,                 #파라미터(x와y에 값을 대입)
    train_size=0.7,      #파라미터 (train, test 중 하나만 지정해도됨)
    #test_size=0.3,      #파라미터
    #shuffle=True,       #파라미터(shuffle=랜덤하게 섞는다, 기본값(default)=true, 데이터를 분할하기 전에 미리 섞어놓는 것을 의미)
    random_state=123     #파라미터(random_state는 호출할 때마다 동일한 학습/테스트용 데이터 세트를 생성하기 위해 주어지는 난수 값이다.
)                        #        train_test_split()는 랜덤으로 데이터를 분리하므로 random_state를 설정하지 않으면 수행할 때마다 다른 학습/테스트 데이터 세트가 생성된다.
                         #        따라서 random_state를 설정하여 수행 시 결과값을 동일하게 맞춰주는 것이다.
                         #        random_state를 지정해줘야 여러번 실행했을 때에도 고정된 값을 얻을 수 있다. 
                         #        default 값은 none이다. none의 경우 여러번 실행하면 다 다른 값이 나온다. (난수값이 다 다르기 때문이다.))      

print('X_train :', X_train)
print('X_test :', X_test)
print('Y_train :', Y_train)
print('Y_test :',Y_test)

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
model.fit(X_train,Y_train,epochs=200, batch_size=1)

#4.평가, 예측
loss=model.evaluate(X_test,Y_test) # 평가(test) 데이터로만 평가해야함 (훈련 데이터 범위 내에서 평가 데이터를 분리)
print('loss : ', loss)
result=model.predict([11])
print('[11]의 결과', result)

'''
loss :  0.045503169298172
[11]의 결과 [[9.876015]]
'''



