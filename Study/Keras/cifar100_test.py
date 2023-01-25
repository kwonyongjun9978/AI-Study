import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar100

# Load the CIFAR-100 dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the CNN architecture
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(100, activation='softmax'))

# Define a loss function and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the network
model.fit(X_train, y_train, epochs=100)

# Evaluate the network on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)