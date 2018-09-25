from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import config
import util
from sklearn.externals import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load the model which will be executed
config.model = 'antnet'
util.set_img_format()
model_module = util.get_model_class_instance()
config.classes = joblib.load(config.get_classes_path())
tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.allow_growth=True
pb_model_path = "pbsave/frozen_test_ant.pb"
sess = tf.Session(config=tfconfig)
# load back the pb file
with gfile.FastGFile(pb_model_path, "rb") as f:
    output_graph_def = tf.GraphDef()
    output_graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(output_graph_def, name="")
    # variable_name = [v.name for v in tf.all_variables()]
    # f = open("pbsave/tensor_restore.txt", "w+")
    # print(output_graph_def, file=f)

# init session
sess.run(tf.global_variables_initializer())

# get the key tensors
input1 = sess.graph.get_tensor_by_name('input_1:0')
input2 = sess.graph.get_tensor_by_name("input_2:0")
classify_dense = sess.graph.get_tensor_by_name("classify_dense/Softmax:0")
embedding_1 = sess.graph.get_tensor_by_name("embedding_1/embedding_lookup:0")

def load_image_into_numpy_array(image, im_width, im_height, input_type):
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(input_type)

# load the image and resize to fixed size, together preprocess it
img = Image.open('pbsave/test.jpg')
img = img.resize((224,224))
input_data = load_image_into_numpy_array(img, 224, 224, np.float32)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

# expand the dim
input_data = np.expand_dims(preprocess_input(input_data), axis=0)
classes_in_keras_format = util.get_classes_in_keras_format()

# prepare the dict and run the session
feed_dict = {input1: input_data, input2: [[1]]}
classify_result, embedding = sess.run([classify_dense, embedding_1], feed_dict=feed_dict)

# print the results
print(classes_in_keras_format.keys(), classes_in_keras_format.values())
print(classify_result, embedding)
#print(list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(np.argmax(classify_result[0], axis=1))])
print(list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(np.argmax(classify_result[0]))])

