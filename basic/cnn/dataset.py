from sklearn.utils import shuffle
import os
import argparse
import numpy as np
import glob

class DataSet():
	def __init__(self, images, labels):
		self._size = images.shape[0]
		self._images = images
		self._labels = labels
		self._done_epoch = 0
		self._index_in_epoch = 0

	@property
	def size(self):
		return self._size
	@property
	def images(self):
		return self._images
	@property
	def labels(self):
		return self._labels
	@property
	def done_epoch(self):
		return self._done_epoch
	@property
	def index_in_epoch(self):
		return self._index_in_epoch

	def next_batch(self, batchSize):
		start = self._index_in_epoch
		self._index_in_epoch += batchSize

		if self._index_in_epoch > self._size:
			self._done_epoch += 1
			start = 0
			self._index_in_epoch = batchSize
			assert batchSize <= self._size
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

def load_set(path, classes):
	images = []
	labels = []

	for entry in classes:
		print('Reading {} class images'.format(entry))

		index = classes.index(entry)
		binaries = glob.glob(os.path.join(path, entry, '*npz'))

		for binary in binaries:
			mode = binary.endswith('labels.npz')
			if mode == 0:
				image = np.load(binary)['storage']
				images = image if index == 0 else np.append(images, image, axis=0) 
			else:
				label = np.load(binary)['storage']
				labels = label if index == 0 else np.append(labels, label, axis=0) 

	image = image.astype(np.float32)
	image = np.multiply(image, 1.0/255.0)

	return images, labels

def read_data_sets(path_to_train_set, path_to_validation_set, classes):
	class DataSets():
		pass

	train_set_images, train_set_labels = load_set(path_to_train_set, classes)
	validation_set_images, validation_set_labels = load_set(path_to_validation_set, classes)

	train_set_images, train_set_labels = shuffle(train_set_images, train_set_labels)
	validation_set_images, validation_set_labels = shuffle(validation_set_images, validation_set_labels)

	data_sets = DataSets()

	data_sets.trainSet = DataSet(train_set_images, train_set_labels)
	data_sets.validationSet = DataSet(validation_set_images, validation_set_labels)

	return data_sets

'''

For testing

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", type=str, help="Path to train set")
	parser.add_argument("--validation", type=str, help="Path to validation set")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	a = read_data_sets(args.train, args.validation)
	print a.trainSet.images.shape, a.validationSet.images.shape

'''