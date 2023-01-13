import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (뒤에 1이 없으니까 흑백) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[1000])
print(y_train[1000])

import matplotlib.pyplot as plt
plt.imshow(x_train[1000], 'gray')
plt.show()
#60000장의 데이터 셋을 가지고 훈련을 한다