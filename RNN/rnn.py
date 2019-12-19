if __name__ == "__main__":
	# Recurrent Neural Network

	import keras
	import tensorflow as tf
	config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
	sess = tf.Session(config=config) 
	keras.backend.set_session(sess)

	# Part 1 - Data Preprocessing
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd

	# Importing the training set
	dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
	training_set = dataset_train.iloc[:, 1:2].values

	# Feature Scaling
	from sklearn.preprocessing import MinMaxScaler
	sc = MinMaxScaler(feature_range= (0,1))
	training_set_scaled = sc.fit_transform(training_set)

	# Creating a a data structure with 60 timesteps and 1 output
	X_train = []
	y_train = []

	# Because we have 60 timesteps, we can only start predicting from the 60th day
	for i in range(60, len(training_set_scaled)):
		X_train.append((training_set_scaled[i-60:i, 0]))
		y_train.append(training_set_scaled[i, 0])

	X_train, y_train = np.array(X_train), np.array(y_train)

	# Reshaping
	# The 1 is the number of indicators
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

	# Part 2 - Building the RNN

	# Importing the Keras libraries and packages
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	from keras.layers import Dropout

	# Initializing the RNN
	regressor = Sequential()

	# Adding the first LSTM layer and some Dropout regularization
	# units; number of neurons in the layer
	# return_sequences; true if we have several LSTM layers
	# input_shape; the last two arguments of np.reshape above
	regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
	regressor.add(Dropout(0.2))

	# Adding a second LSTM layer and some Dropout regularization
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# Adding a third LSTM layer and some Dropout regularization
	regressor.add(LSTM(units = 50, return_sequences = True))
	regressor.add(Dropout(0.2))

	# Adding a fourth LSTM layer and some Dropout regularization
	regressor.add(LSTM(units = 50))
	regressor.add(Dropout(0.2))

	# Adding the outout layer
	regressor.add(Dense(units = 1))

	# Compiling the RNN
	regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

	# Fitting the RNN to the Training set
	regressor.fit(X_train, y_train, epochs = 100, batch_size=32)

	# Part 3 - Making the predictions and visualizing the results

	# Getting the real stock price of 2017
	dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
	real_stock_price = dataset_test.iloc[:, 1:2].values

	# Getting the predicted stock price of 2017
	dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
	inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
	inputs = inputs.reshape(-1, 1)
	inputs = sc.transform(inputs)
	X_test = []
	for i in range(60,80):
		X_test.append(inputs[i-60:i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	predicted_stock_price = regressor.predict(X_test)
	predicted_stock_price = sc.inverse_transform(predicted_stock_price)
	
	regressor.save("rnn_regressor.h5")
	print("Saved model to disk")

	# Eror Metric
	import math
	from sklearn.metrics import mean_squared_error
	rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
	norm_rmse = rmse/(sum(real_stock_price)/len(real_stock_price))

	# Visualizing the results
	plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
	plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
	plt.title("Google Stock Price Prediction")
	plt.xlabel("Time")
	plt.ylabel("Price")
	plt.legend()
	plt.show()