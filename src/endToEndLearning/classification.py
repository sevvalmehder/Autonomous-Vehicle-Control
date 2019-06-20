# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import load_data
import numpy as np
from model import nvidia_model



class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 20 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save_weights("model_{}.h5".format(epoch))

def classify_labels(labels):
	'''
	Divide steering angle to the 19 class
	'''

	new_labels = []

	for label in labels:

		if label >= 0:
			new_label = int(label/18)

		elif label >= -18:
			new_label = 0

		else:
			label *= -1
			new_label = int(label/18)
			new_label += 9

		new_labels.append(new_label)

	return np.array(new_labels)

def main():

	# Load the dataset
	train_images, train_labels, test_images, test_labels = load_data.return_data("driving_dataset/train")
	train_labels = classify_labels(train_labels)
	test_labels = classify_labels(test_labels)

	# Load model
	model = nvidia_model()


	model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	saver = CustomSaver()
	model.fit(train_images, train_labels, callbacks=[saver], epochs=60, validation_data=(test_images,test_labels))

	test_loss, test_acc = model.evaluate(test_images, test_labels)
	print('Test accuracy:', test_acc)

	model.save_weights("model_last.h5")

if __name__ == '__main__':
	main()