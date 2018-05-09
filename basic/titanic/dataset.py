import csv
import numpy
from sklearn.utils import shuffle

class DataSet():
	def __init__(self, data, labels):
		self._size = data.shape[0]
		self._data = data
		self._labels = labels
		self._done_epoch = 0
		self._index_in_epoch = 0

	@property
	def size(self):
		return self._size
	@property
	def data(self):
		return self._data
	@property
	def labels(self):
		return self._labels
	@property
	def done_epoch(self):
		return self._done_epoch
	@property
	def index_in_epoch(self):
		return self._index_in_epoch

	def next_batch(self, batch_size):
		assert batch_size <= self._size

		start = self._index_in_epoch
		end = start + batch_size

		if end >= self._size:
			self._done_epoch += 1
			start = 0
			end = batch_size

		self._index_in_epoch = end
		return self._data[start:end], self._labels[start:end]

def normalize(x):
	x = numpy.array(x, dtype=numpy.float)
	
	min_value = numpy.amin(x, axis=0)
	max_value = 1 / numpy.float(numpy.amax(x, axis=0) - min_value)
	x = (x - min_value) * max_value
	return x

def load_set():
	file = open('titanic.csv', 'r')
	reader = csv.reader(file)

	labels = []
	ranks = []
	sexs = []
	ages = []
	siblings = []
	parents = []

	for row in reader:
		# Collect correct answers
		label = numpy.zeros(2)
		label[int(row[0])] = 1.0
		labels.append(label)
		# And other data
		ranks.append(int(row[1]) % 3)
		sexs.append(0 if row[3] == 'male' else 1)
		ages.append(float(row[4]))
		siblings.append(int(row[5]))
		parents.append(int(row[6]))

	ranks = normalize(ranks)
	sexs = normalize(sexs)
	ages = normalize(ages)
	siblings = normalize(siblings)
	parents = normalize(parents)

	data = numpy.column_stack((ranks, sexs, ages, siblings, parents))
	data, labels = shuffle(data, labels)
	return data, labels

def read_data_sets():
	class DataSets():
		pass

	data_sets = DataSets()

	data, labels = load_set()

	data_sets.train_set = DataSet(data[int(len(data)*0.8):], labels[int(len(labels)*0.8):])
	data_sets.validation_set = DataSet(data[:int(len(data)*0.8)], labels[:int(len(labels)*0.8)])

	return data_sets
