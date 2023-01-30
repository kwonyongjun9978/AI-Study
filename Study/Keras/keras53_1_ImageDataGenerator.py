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

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    #Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size=(200,200),
    batch_size=10,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
    #Found 120 images belonging to 2 classes.
)

print(xy_train) #<keras.preprocessing.image.DirectoryIterator object at 0x00000181F09EE040>

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)

# print(xy_train[0])
# print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape)  # (10, 200, 200, 1)
print(xy_train[0][1].shape)  # (10,)
print(xy_train[15][0].shape)  
print(xy_train[15][1].shape)

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
print(type(xy_train[0][1])) #<class 'numpy.ndarray'>





