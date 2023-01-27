from keras.layers import LSTM, Dense
from keras.models import Sequential
import numpy as np

# Assume that your data is stored in a 2D array called 'data'
# where the first column is the date and the second column is the stock price

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Split the data into training and test sets
train_data = data[:int(data.shape[0] * 0.8)]
test_data = data[int(data.shape[0] * 0.8):]

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, input_shape=(train_data.shape[1], 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Reshape the data for the LSTM
train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))

# Train the model
model.fit(train_data, epochs=100, batch_size=1, verbose=2)

# Make predictions on the test data
predictions = model.predict(test_data)