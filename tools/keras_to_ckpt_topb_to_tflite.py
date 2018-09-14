import tensorflow as tf
import keras
from keras import backend as K
# 1111111111111111111111111111111111111
# Add ops to save and restore all the variables.
def center_loss_fn(y_true, y_pred):
    # l2_loss layer output is the y_pred, the value will be passed to the optimizer then
    return y_pred
# save the model with custom_objects
model = keras.models.load_model("trained/model-antnet.h5",custom_objects={'center_loss_fn': center_loss_fn})
sess = K.get_session()
saver = tf.train.Saver()
save_path = saver.save(sess, "pbsave/model.ckpt")
# save the pbtxt in order to check the output things in the tensorboard
tf.train.write_graph(sess.graph_def, 'pbsave', 'graph.pbtxt')


# 2222222222222222222222222222222222222
# after the ckpt and graph.pbtxt, then use the below cmd to freeze.
# freeze_graph    \
# --input_graph=pbsave/graph.pbtxt    \
# --input_checkpoint=pbsave/model.ckpt   \
# --input_binary=false   \
# --output_graph=pbsave/frozen_test.pb    \
# --output_node_names=classify_dense/Softmax,embedding_1/embedding_lookup


# 33333333333333333333333333333333333333
# then the below codes help us to export the tflite
graph_def_file = "pbsave/frozen_test.pb"
input_arrays = ["input_2","input_3"]
output_arrays = ["classify_dense/Softmax","embedding_1/embedding_lookup"]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


# the extra part , which we can use shell not 33333333333333333333333333333333
# bazel run -c opt tensorflow/contrib/lite/toco:toco --  \
#     --input_file=/home/hdd300g/modelers/modeler001/share/tf-models/research/hequndata-cow/frozen_test.pb \
#     --output_file=/home/hdd300g/modelers/modeler001/share/tf-models/research/hequndata-cow/frozen_test.tflite  \
#     --input_shapes=1,224,224,3:1,1  \
#     --input_data_types=FLOAT,FLOAT \
#     --input_arrays=input_1,input_2  \
#     --output_arrays='classify_dense/Softmax','embedding_1/embedding_lookup'   \
#     --inference_type=FLOAT  \
#     --change_concat_input_ranges=false \
#     --allow_custom_ops
