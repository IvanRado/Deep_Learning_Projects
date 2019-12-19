if __name__ == "__main__":
	## Convolutional Neural Network
	
	
	import keras
	import tensorflow as tf
	config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
	sess = tf.Session(config=config) 
	keras.backend.set_session(sess)

	# Importing the Keras libraries and packages
	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
	 
	# Initialising the CNN
	classifier = Sequential()
	 
	# Step 1 - Convolution
	# no of filters/ feature detector/ feature maps - 32
	# its dimensions within (rows, cols)
	# input_shape - to transform all images to standard format (width, height, channel)
	# activation function used for more non linearity
	classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	 
	# Adding a second convolutional layer
	classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	 
	classifier.add(Conv2D(64, (3 ,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	 
	classifier.add(Conv2D(128, (3 ,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2, 2)))
	 
	# Step 3 - Flattening
	classifier.add(Flatten())
	 
	# Step 4 - Full connection
	classifier.add(Dropout(0.6))
	classifier.add(Dense(units = 128, activation = 'relu'))
	classifier.add(Dropout(0.3))
	classifier.add(Dense(units = 1, activation = 'sigmoid'))
	 
	print('\n compiling the CNN');
	# Compiling the CNN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	 
	print('\n Fitting the CNN to dataset')
	# Part 2 - Fitting the CNN to the images
	from keras.preprocessing.image import ImageDataGenerator
	train_datagen = ImageDataGenerator(rescale = 1./255,
									   shear_range = 0.2,
									   zoom_range = 0.2,
									   horizontal_flip = True)
	test_datagen = ImageDataGenerator(rescale = 1./255)
	 
	training_set = train_datagen.flow_from_directory('dataset/training_set',
													 target_size = (256, 256),
													 batch_size = 32,
													 class_mode = 'binary')
	test_set = test_datagen.flow_from_directory('dataset/test_set',
												target_size = (256, 256),
												batch_size = 32,
												class_mode = 'binary')
	 
	classifier.fit_generator(training_set,
							 steps_per_epoch = 8000,
							 epochs = 10,
							 validation_data = test_set,
							 validation_steps = 2000)
	 
	classifier.save("cnn_classifier.h5")
	print("Model saved to disk")