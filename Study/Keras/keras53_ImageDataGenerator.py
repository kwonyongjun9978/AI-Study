import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#이미지 데이터를 증폭시킨다
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

#테스트 데이터는 rescale만 한다 :
test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_datagen.flow_from_directory()