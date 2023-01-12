from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten 
#2D(2차원,평면,그림,이미지)

model=Sequential()
'''
CNN
독립변수 데이터(x) : 이미지
이미지를 일정하게 조각내서 조각들의 특성값들을 계산해서 종속변수(y)를 알아내는 작업 
'''
model.add(Conv2D(filters=10, kernel_size=(2,2),
                 input_shape=(5,5,1))) 
#(5,5)크기의 1개의 filters를 가지고있는 이미지를 (2,2)크기로 조각낸다 -> (4,4)으로 변경된 이미지를 10개(filters)로 변경(4,4,10)
model.add(Conv2D(filters=5, kernel_size=(2,2))) #(3,3,5)
model.add(Flatten()) #45(3x3x5)
model.add(Dense(10))
model.add(Dense(1)) 

model.summary()
#filters의 개수 = 그다음 output 노드의 개수
#하이퍼파라미터 튜닝 : filters ,kernel_size 조절 + activation 선택
