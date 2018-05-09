import tensorflow as tf
import numpy as np

# dataset
sequence = np.arange(0, 10)
true_sequence = np.arange(0, -10, -1)

test_sequence = np.arange(100, 200)
test_true_sequence = np.arange(-100, -200, -1)

# init
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
w = tf.Variable(0.0, name='weight')
init = tf.global_variables_initializer()

# model
y = w * x

# loss function
#loss = tf.losses.mean_squared_error(labels=y_, predictions=y)
loss = tf.reduce_sum((y_ - y) ** 2) / tf.cast(tf.size(y_), tf.float32)

#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as session:
    session.run(init)
    for i in range(180):
        feed_dict = {x:sequence, y_:true_sequence}
        session.run(optimizer, feed_dict)
        if i % 10 == 0:
            feed_dict = {x:test_sequence, y_:test_true_sequence}
            print "Loss: {0[0]:.4f} ~ Weight: {0[1]:.4f}".format(session.run([loss, w], feed_dict))
