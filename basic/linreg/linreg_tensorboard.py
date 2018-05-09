import numpy as np
import tensorflow as tf

# summary writer
writer = tf.summary.FileWriter('/tmp/logs')

# dataset
sequence = np.arange(0, 10) # [0 1 2 3 4 5 6 7 8 9]
true_sequnce = map(lambda entry: entry * 2 + 1, sequence) # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

test_sequence = np.arange(100, 200)
test_true_sequence = map(lambda entry: entry * 2 + 1, test_sequence)

# init
x = tf.placeholder(tf.float32, name='x')
y_true = tf.placeholder(tf.float32, name='labels')

w = tf.Variable(0.0, tf.float32, name='weight')
b = tf.Variable(0.0, tf.float32, name='bias')
init = tf.global_variables_initializer()

# model
with tf.name_scope('model'):
    y = w * x + b

# success rate
with tf.name_scope('loss'):
    loss = tf.reduce_sum((y - y_true) ** 2) / tf.cast(tf.size(y), tf.float32)

# optimizer
with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.003).minimize(loss)

# tensorboard stuff
tf.summary.scalar('loss', loss)
tf.summary.histogram('weight', w)
tf.summary.histogram('biases', b)
merged_summary = tf.summary.merge_all()

with tf.Session() as session:
    session.run(init)
    writer.add_graph(session.graph)
    for i in range(5000):
            feed_dict = {x: sequence, y_true: true_sequnce}
        session.run(optimizer, feed_dict)
        if i % 10 == 0:
            feed_dict = {x: test_sequence, y_true: test_true_sequence}
            print 'Loss: {0[0]:.4f} ~ Weight: {0[1]:.4f} ~ Bias: {0[2]:.4f}'.format(session.run([loss, w, b], feed_dict))
            summary = session.run(merged_summary, feed_dict)
            writer.add_summary(summary, i)
