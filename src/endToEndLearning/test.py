from keras.models import load_model

import load_data

import classification 

from operator import truediv
from sklearn import metrics
import numpy as np

from model import NVIDA, nvidia_model
import argparse



def AA_andEachClassAccuracy(confusion_matrix):
	counter = confusion_matrix.shape[0]
	list_diag = np.diag(confusion_matrix)
	list_raw_sum = np.sum(confusion_matrix, axis=1)
	each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
	average_acc = np.mean(each_acc)
	return each_acc, average_acc

def test(args):
	print("start")
	# Load the model
	model = nvidia_model()
	model.load_weights(args.model)
	print("model loaded")

	# Load the test images
	test_images, test_labels, _, _ = load_data.return_data(args.path, 1)
	labels = classification.classify_labels(test_labels)

	# Make prediction for every test image
	predictions = []
	for image in test_images:
		image = np.array([image])
		pred = model.predict(image)
		predictions.append(np.argmax(pred[0]))

	# Calculate matrics
	# Overall classification accuracy
	overall_acc = metrics.accuracy_score(predictions, labels)
	# Kappa value
	kappa = metrics.cohen_kappa_score(predictions, labels)
	# Confusion matrix
	confusion_matrix = metrics.confusion_matrix(predictions, labels)
	# Classification accuracy of every class and avarage accuracy
	each_acc, average_acc = AA_andEachClassAccuracy(confusion_matrix)

	# Print the results
	print("The overall accuracy: {}\tThe kappa value: {}\t".format(overall_acc, kappa))

if __name__ == '__main__':
	print("start")
	argparser = argparse.ArgumentParser()
	argparser.add_argument(
		'-p', '--path',
		default='driving_dataset/test',
		help='the path of test images')
	argparser.add_argument(
		'-m', '--model',
		default='model.h5',
		help = 'the model to test')
	args = argparser.parse_args()

	test(args)