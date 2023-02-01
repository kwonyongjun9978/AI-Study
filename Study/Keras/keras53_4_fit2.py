import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(100,100),
    batch_size=10,   #lne()
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/', 
    target_size=(100,100),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    #Found 120 images belonging to 2 classes.
)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(128, (3,3), input_shape=(100,100,1), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['acc'])

# hist = model.fit_generator(xy_train, steps_per_epoch=16, epochs=100,
#                     validation_data=xy_test,
#                     validation_steps=4, )

hist = model.fit( # xy_train[0][0], xy_train[0][1],
                 xy_train, 
                  # batch_size=16,
                 epochs=10, 
                 validation_data=(xy_test[0][0], xy_test[0][1]),
                  # validation_split=0.2
                )

#4. 평가, 예측
accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1]) #loss[100]
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])
# results=model.evaluate(xy_test)
# print('loss, acc : ', results)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #판사이즈(figsize(가로길이, 세로길이) 단위는 inch이다.)
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid() #격자
plt.xlabel('epochs') #x축 라벨
plt.ylabel('loss')   #y축 라벨
plt.title('ImageDataGenerator2') #제목 이름 표시
plt.legend() #선의 이름
#plt.legend(loc='upper left') 
plt.show()

'''
loss :  1.2579003616508544e-08
val_loss :  7.601393699645996
accuracy :  1.0
val_acc :  0.5416666865348816
'''








