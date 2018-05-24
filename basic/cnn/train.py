import dataset as ds
import tensorflow as tf
import numpy as np
import argparse as ap
import os

# Pre-init
tf.set_random_seed(0)
writer = tf.summary.FileWriter('/tmp/logs')

# Init
classes = ['back', 'empty', 'front', 'side']
S = len(classes)
batch_size = 75
H = 300
W = 225
C = 3

with tf.name_scope('inputs'):
	# Image array like 4D vector [batch_size, h, w, c]
	x = tf.placeholder(tf.float32, shape=[None, H, W, C], name='input')
	# One hot encoded array of correct answers like [batch_size, sizeof_classes]
	y_true = tf.stop_gradient(tf.placeholder(tf.float32, shape=[None, S], name='correct_answers'))
	# Placeholder for step decrease during training
	dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 4 softmax neurons)
K = 8  # first convolutional layer output depth
L = 16  # second convolutional layer output depth
M = 32  # third convolutional layer
N = 100  # fully connected layer

with tf.name_scope('weights'):
	W1 = tf.Variable(tf.truncated_normal([5, 5, C, K], stddev=0.1), name='weight_1')
	W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name='weight_2')
	W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name='weight_3')
	W4 = tf.Variable(tf.truncated_normal([6 * 5 * M, N], stddev=0.1), name='weight_4')
	W5 = tf.Variable(tf.truncated_normal([N, S], stddev=0.1), name='weight_5')

with tf.name_scope('biases'):
	B1 = tf.Variable(tf.truncated_normal([K], stddev=0.1)/S, name='bias_1')
	B2 = tf.Variable(tf.truncated_normal([L], stddev=0.1)/S, name='bias_2')
	B3 = tf.Variable(tf.truncated_normal([M], stddev=0.1)/S, name='bias_3')
	B4 = tf.Variable(tf.truncated_normal([N], stddev=0.1)/ S, name='bias_4')
	B5 = tf.Variable(tf.truncated_normal([S], stddev=0.1)/S, name='bias_5')

with tf.name_scope('model'):
	# Output 100x75
	L1 = tf.nn.dropout(tf.layers.max_pooling2d(tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='SAME') + B1, name='L1'), 2, 3), dropout_prob)
	# Output 33x25
	L2 =  tf.nn.dropout(tf.layers.max_pooling2d(tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2, name='L2'), 3, 3), dropout_prob)
	# Output 10x5
	L3 = tf.nn.dropout(tf.layers.max_pooling2d(tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3, name='L3'), 5, 5), dropout_prob)

	LF = tf.reshape(L3, shape=[-1, 6 * 5 * M], name='LF')

	L4 = tf.nn.dropout(tf.nn.relu(tf.matmul(LF, W4) + B4, name='L4'), dropout_prob)

	Logits = tf.matmul(L4, W5) + B5
	y_pred = tf.nn.softmax(Logits)

with tf.name_scope('evalution_metrics'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=Logits, name='cross_entropy')
	cross_entropy = tf.reduce_mean(cross_entropy)

	acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))
	loss = cross_entropy

with tf.name_scope('optimizer'):
	step = tf.Variable(0, trainable=False)
	start_learning_rate = 0.002
	decay_steps = 2500 # 21 epoch
	decayed_rate = tf.train.exponential_decay(start_learning_rate, step, decay_steps, 0.98, staircase=True)
	optimizer = tf.train.AdamOptimizer(decayed_rate).minimize(cross_entropy, global_step=step)

# Tensorboard stuff

tf.summary.scalar('loss', loss)
tf.summary.scalar('acc', acc)
tf.summary.histogram('weight_1', W1)
tf.summary.histogram('weight_2', W2)
tf.summary.histogram('weight_3', W3)
tf.summary.histogram('weight_4', W4)
tf.summary.histogram('weight_5', W5)
tf.summary.histogram('bias_1',  B1)
tf.summary.histogram('bias_2',  B2)
tf.summary.histogram('bias_3',  B3)
tf.summary.histogram('bias_4',  B4)
tf.summary.histogram('bias_5',  B5)
merged_summary = tf.summary.merge_all()

def train(max_iter, data, save_dir, graph_dir):
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		writer.add_graph(session.graph)

		current_epoch = -1

		dropout_p = 0.85

		for i in range(max_iter):
			# Train
			if i > 400:
				dropout_p = 0.80
			if i > 800:
				dropout_p = 0.75
			if i > 1200:
				dropout_p = 0.70
			if i > 1600:
				dropout_p = 0.65
			if i > 2000:
				dropout_p = 0.60

			x_batch, y_true_batch = data.trainSet.next_batch(batch_size)
			feed_dict = {x: x_batch, y_true: y_true_batch, dropout_prob: dropout_p}
			session.run(optimizer, feed_dict)
			if current_epoch != data.trainSet.done_epoch:
				current_epoch += 1
				print ('Epoch: {0}'.format(data.trainSet.done_epoch))
			# Validate
			if i % 20 == 0 :
				x_batch, y_true_batch = data.validationSet.next_batch(batch_size*4)
				feed_dict = {x: x_batch, y_true: y_true_batch, dropout_prob: 1.0}
				print ('Loss: {0[0]:.4f} ~ Acc: {0[1]:.4f} '.format(session.run([loss, acc], feed_dict)))
				summary = session.run(merged_summary, feed_dict)
				writer.add_summary(summary, i)
				saver.save(session, save_dir)
				tf.train.write_graph(session.graph, graph_dir, 'train.pb', as_text=False)


def parse_args():
	parser = ap.ArgumentParser()
	parser.add_argument("--train", type=str, help="Path to train set")
	parser.add_argument("--validation", type=str, help="Path to validation set")
	parser.add_argument("--save_dir", type=str, help="Path to folder where to save checkpoint data")
	parser.add_argument("--graph_dir", type=str, help="Path to folder where to save checkpoint data")
	parser.add_argument("--max_iter", type=int, help="Path to folder where to save checkpoint data")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	data = ds.read_data_sets(args.train, args.validation, classes)
	train(args.max_iter, data, args.save_dir, args.graph_dir)
