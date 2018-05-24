import tfcoreml as tf_converter
import sys,argparse

parser = argparse.ArgumentParser()
parser.add_argument("--frozen_model", default="", type=str, help="Frozen model file to import")
parser.add_argument("--mlmodel", default="", type=str, help="Mlmodel to output")
args = parser.parse_args()

tf_converter.convert(tf_model_path = args.frozen_model,
                    mlmodel_path = args.mlmodel,
                    output_feature_names = ['model/Softmax:0'],
	                input_name_shape_dict =  {'inputs/input:0' : [1, 300, 225, 3]},
                    image_input_names = 'inputs/input:0',
                    class_labels = 'labels.txt',
                    image_scale = 1.0/255.0)
