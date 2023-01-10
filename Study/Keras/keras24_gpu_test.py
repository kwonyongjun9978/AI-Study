import tensorflow as tf
print(tf.__version__) #2.7.4

gpus=tf.config.experimental.list_physical_devices('GPU')
print(gpus)
#nvidia gpu만 잡힌다

if(gpus):
    print("GPU O")
else:
    print("GPU X") 