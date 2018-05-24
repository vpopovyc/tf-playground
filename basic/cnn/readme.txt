~~~~~~ Freeze graph ~~~~~~

python freeze_graph.py \
--input_graph=graph/train.pb \
--input_binary=True \
--input_checkpoint=checkpoint_dir/ucar.chkp \
--output_graph=graph/frozen_train.pb \
--output_node_names='model/Softmax'

~~~~~~ Conevert python model to coreml model ~~~~~~

python convert.py \
--frozen_model=graph/frozen_train.pb \
--mlmodel=graph/ucar_0_1.mlmodel


