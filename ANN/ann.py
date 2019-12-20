if __name__ == "__main__":
	# To verify GPU is being used for computation
	# from tensorflow.python.client import device_lib
	# print(device_lib.list_local_devices())


	# from keras import backend as K
	# K.tensorflow_backend._get_available_gpus()

	# To enable GPU after importing keras
	import keras
	import tensorflow as tf


	config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
	sess = tf.Session(config=config) 
	keras.backend.set_session(sess)

	# Import Libraries
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd

	# Importing the dataset
	dataset = pd.read_csv('Churn_Modelling.csv')
	X = dataset.iloc[:, 3:13].values
	y = dataset.iloc[:, 13].values

	# Encoding categorical data
	from sklearn.preprocessing import LabelEncoder, OneHotEncoder

	# Make countries categorical
	labelencoder_X_1 = LabelEncoder()
	X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

	# Make gender categorical
	labelencoder_X_2 = LabelEncoder()
	X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

	# Conver the countries column into dummy variables
	onehotencoder = OneHotEncoder(categorical_features = [1])
	X = onehotencoder.fit_transform(X).toarray()

	# Remove one of the dummy variables to avoid the dummy variable trap
	X = X[:, 1:]

	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)


	from keras.wrappers.scikit_learn import KerasClassifier
	from sklearn.model_selection import GridSearchCV
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout
	def build_classifier(optimizer, hidden_neurons):
		classifier = Sequential()
		classifier.add(Dense(units = hidden_neurons, activation = 'relu', input_dim = 11, kernel_initializer = 'uniform'))
		classifier.add(Dense(units = hidden_neurons, kernel_initializer = 'uniform', activation = 'relu'))
		classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
		classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
		return classifier

	classifier = KerasClassifier(build_fn = build_classifier)
	parameters = {'batch_size': [40, 80], 
				  'epochs': [300, 600],
				  'optimizer': ['adam', 'rmsprop'],
				  'hidden_neurons': [9, 10]}
	grid = GridSearchCV(estimator = classifier, 
						param_grid = parameters, 
						scoring = 'accuracy',
						cv = 10)

	grid_search = grid.fit(X_train, y_train)
	best_parameters = grid_search.best_params_
	
	classifier = build_classifier(best_parameters['optimizer'], best_parameters['hidden_neurons'])
	classifier.fit(X_train, y_train, epochs = best_parameters['epochs'], batch_size = best_parameters['batch_size'])

	classifier.save("classifier.h5")
	print("Saved model to disk")
	
	
	print("Best parameters: {}".format(best_parameters))
	best_accuracy = grid_search.best_score_
	print("Best accuracy: {}".format(best_accuracy))



	# Can load model with model = load_model(filename)