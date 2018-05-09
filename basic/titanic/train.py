import dataset as ds
import tensorflow as tf
'''
Loss: 0.5063 ~ Acc: 0.7983
Loss: 0.5056 ~ Acc: 0.7997
Loss: 0.5052 ~ Acc: 0.8025

two convolutional layers => flatten layer => fully connected layer
'''

data_sets = ds.read_data_sets()
writer = tf.summary.FileWriter('/tmp/logs')

with tf.name_scope('inputs'):
	x = tf.placeholder(tf.float32, [None, 5])
	y_true = tf.placeholder(tf.float32, [None, 2])
	data = tf.reshape(x, [-1, 5, 1])

with tf.name_scope('vars'):
	K1 = tf.Variable(tf.truncated_normal([3, 1, 4], dtype=tf.float32, stddev=0.1))
	K2 = tf.Variable(tf.truncated_normal([4, 4, 8], dtype=tf.float32, stddev=0.1))
	WC = tf.Variable(tf.truncated_normal([8, 2], dtype=tf.float32, stddev=0.1))
	B1 = tf.Variable(tf.ones([4]))
	B2 = tf.Variable(tf.ones([8]))
	BC = tf.Variable(tf.ones([2]))

with tf.name_scope('model'):
	L1 = tf.nn.relu(tf.nn.conv1d(data, K1, 1, 'SAME') + B1)
	L2 = tf.nn.relu(tf.nn.conv1d(L1, K2, 5, 'SAME') + B2)
	LF = tf.reshape(L2, [-1, 8])
	LC = tf.matmul(LF, WC) + BC
	y_pred = tf.nn.softmax(LC)

with tf.name_scope('evaluation_metrics'):
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred, name='cross_entropy')
	loss = tf.reduce_mean(cross_entropy)
	acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))

with tf.name_scope('optimizer'):
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.01
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10000, 0.96, staircase=True)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

with tf.name_scope('tensorboard_stuff'):
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('acc', acc)
	tf.summary.histogram('kernel_1', K1)
	tf.summary.histogram('kernel_2', K2)
	tf.summary.histogram('weight_fully_connected', WC)
	tf.summary.histogram('bias_1',  B1)
	tf.summary.histogram('bias_2',  B2)
	tf.summary.histogram('bias_fully_connected',  BC)
	merged_summary = tf.summary.merge_all()

with tf.Session() as session:
	session.run(tf.global_variables_initializer())

	for i in range(10000):
		x_batch, y_batch = data_sets.train_set.next_batch(25)
		feed_dict = {x: x_batch, y_true: y_batch}

		session.run(optimizer, feed_dict)
		if i % 100 == 0:
			x_batch, y_batch = data_sets.validation_set.next_batch(data_sets.validation_set.size)
			feed_dict = {x: x_batch, y_true: y_batch}
			print 'Loss: {0[0]:.4f} ~ Acc: {0[1]:.4f}'.format(session.run([loss, acc], feed_dict))
			summary = session.run(merged_summary, feed_dict)
			writer.add_summary(summary, i)
