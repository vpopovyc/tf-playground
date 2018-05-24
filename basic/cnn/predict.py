import tensorflow as tf
import dataset as ds
import argparse as ap

def parse_args():
	parser = ap.ArgumentParser()
	parser.add_argument("--train", type=str, help="Path to train set")
	parser.add_argument("--validation", type=str, help="Path to validation set")
	parser.add_argument("--chkp", type=str, help="Path to folder where to save checkpoint data")
	parser.add_argument("--chkp_dir", type=str, help="Path to folder where to save checkpoint data")
	return parser.parse_args()

def graph_names(graph):
	[n.name for n in tf.get_default_graph().as_graph_def().node]

def test(dataset, checkpoint, checkpoint_dir):
	sess = tf.Session()
	saver = tf.train.import_meta_graph(checkpoint)
	saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
	graph = tf.get_default_graph()
	
	input_node = graph.get_tensor_by_name('inputs/input:0')
	dropp_node = graph.get_tensor_by_name('inputs/dropout_prob:0')
	predc_node = graph.get_tensor_by_name('model/Softmax:0')

	input_batch, true_batch = dataset.validationSet.next_batch(1)

	feed_dict = {input_node: input_batch, dropp_node: 1.0}

	result = sess.run(predc_node, feed_dict)

	print(result)
	print(true_batch)

if __name__ == "__main__":
	args = parse_args()
	data = ds.read_data_sets(args.train, args.validation, ['back', 'empty', 'front', 'side'])
	test(data, args.chkp, args.chkp_dir)
