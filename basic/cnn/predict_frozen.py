import tensorflow as tf
import dataset as ds
import argparse as ap

def parse_args():
	parser = ap.ArgumentParser()
	parser.add_argument("--frozen_model", type=str, help="Path to train set")
	parser.add_argument("--train", type=str, help="Path to train set")
	parser.add_argument("--validation", type=str, help="Path to validation set")
	return parser.parse_args()

def graph_names(graph):
	for op in graph.get_operations():
		print(op.name)

def test(dataset, frozen_model):
	with tf.gfile.GFile(frozen_model, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	with tf.Graph().as_default() as graph:
		tf.import_graph_def(graph_def, name="ucar_cnn_0_1")


	sess = tf.Session(graph=graph)
	graph_names(graph)

	input_node = graph.get_tensor_by_name('ucar_cnn_0_1/inputs/input:0')
	# dropp_node = graph.get_tensor_by_name('ucar_cnn_0_1/inputs/dropout_prob:0')
	predc_node = graph.get_tensor_by_name('ucar_cnn_0_1/model/Softmax:0')

	input_batch, true_batch = dataset.validationSet.next_batch(1)

	feed_dict = {input_node: input_batch}

	result = sess.run(predc_node, feed_dict)

	print(result)
	print(true_batch)

if __name__ == "__main__":
	args = parse_args()
	data = ds.read_data_sets(args.train, args.validation, ['back', 'empty', 'front', 'side'])
	test(data, args.frozen_model)
