from __future__ import print_function
from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]
        
# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('./graph/frozen_train_without_dropout.pb', 'rb') as f:
    data = f.read()
    graph.ParseFromString(data)
    
display_nodes(graph.node)

# Connect 'MatMul_1' with 'Relu_2'
# graph.node[50].input[0] = 'model/L4' # 44 -> MatMul_1
# # Remove dropout nodes
# nodes = graph.node[:39] + graph.node[50:] # 33 -> MatMul_1 

display_nodes(graph.node)

# Save graph
output_graph = graph_pb2.GraphDef()
output_graph.node.extend(graph.node)
with tf.gfile.GFile('./graph/frozen_train_without_dropout.pb', 'w') as f:
    f.write(output_graph.SerializeToString())

