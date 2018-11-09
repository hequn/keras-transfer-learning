from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import config
import util
from sklearn.externals import joblib
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load the model which will be executed
config.model = 'xception'
util.set_img_format()
model_module = util.get_model_class_instance()
config.classes = joblib.load(config.get_classes_path())
tfconfig = tf.ConfigProto(allow_soft_placement=True)
# tfconfig.gpu_options.allow_growth=True
pb_model_path = "pbsave/frozen_test_embbeding.pb"
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
# input2 = sess.graph.get_tensor_by_name("input_2:0")
classify_dense = sess.graph.get_tensor_by_name("classify_dense/Softmax:0")
# embedding_centor = sess.graph.get_tensor_by_name("embedding_1/embedding_lookup:0")
embedding_feature = sess.graph.get_tensor_by_name("embedding/Elu:0")


def load_image_into_numpy_array(image, im_width, im_height, input_type):
    return np.array(image).reshape(
        (im_height, im_width, 1)).astype(input_type)


def contrast_demo(img1, c=1, b=10):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channel = img1.shape
    blank = np.zeros([rows, cols, channel], img1.dtype)
    np.zeros(img1.shape, dtype=np.uint8)
    dst = cv2.addWeighted(img1, c, blank, 1 - c, b)
    cv2.imshow("con_bri_demo", dst)
    cv2.waitKey(1000)
    return dst


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq


def gamma_trans(img, gamma):
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(img, gamma_table)


# load the image and resize to fixed size, together preprocess it
img = cv2.imread('pbsave/140134_2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Laplacian(img, cv2.CV_32F, ksize=3, scale=1, delta=0)
img = cv2.resize(img, (224, 224))
# img = contrast_demo(img,c=1,b=5)
# img = hisEqulColor(img)
img = gamma_trans(img, 0.8)
cv2.imshow("con_bri_demo1", img)
cv2.waitKey(1000)
input_data1 = load_image_into_numpy_array(img, 224, 224, np.float32)

# load the image and resize to fixed size, together preprocess it
img = cv2.imread('pbsave/140134_4.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.Laplacian(img, cv2.CV_32F, ksize=3, scale=1, delta=0)
img = cv2.resize(img, (224, 224))
# img = contrast_demo(img,c=1,b=5)
# img = hisEqulColor(img)
img = gamma_trans(img, 0.8)
cv2.imshow("con_bri_demo2", img)
cv2.waitKey(1000)
input_data2 = load_image_into_numpy_array(img, 224, 224, np.float32)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


# expand the dim
input_data1 = np.expand_dims(preprocess_input(input_data1), axis=0)
input_data2 = np.expand_dims(preprocess_input(input_data2), axis=0)
input_data = np.vstack((input_data1, input_data2))
classes_in_keras_format = util.get_classes_in_keras_format()

# prepare the dict and run the session
feed_dict = {input1: input_data}
classify_result, embedding_f = sess.run([classify_dense, embedding_feature],
                                        feed_dict=feed_dict)


# print the results
# print(classes_in_keras_format.keys(), classes_in_keras_format.values())
# print(classify_result)
# print(embedding_f[0])

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1)) - 2 * np.dot(x1, x2) + np.sum(np.square(x2)))


# cosine
def cosin_distance(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


print('euclidean distance is: ', euclidean_distance(embedding_f[0], embedding_f[1]))
print('euclidean distance is: ', cosin_distance(embedding_f[0], embedding_f[1]))
