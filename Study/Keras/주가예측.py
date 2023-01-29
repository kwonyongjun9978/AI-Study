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

# 01 ~ 14까지 Conv1D로 만들어
# 삼성전자와 아모레 주가를 앙상블 모델로 만들어 삼성전자 주가 예측하기
# (컬럼 5개 이상 쓰기, 삼성 월요일 시가 예측해보기) ->소스 ,가중치 제출
#양식
#제목 : 권용준 00,000원