from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential, Model
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2


def load_image(path, color_mode='bgr'):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224), color_mode=color_mode)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    x = preprocess_input(x)
    return x


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def visualize_class_activation_map(model_input, image, target_class, lay_name):
    model = model_input
    original_img = image
    width, height, _ = original_img[0].shape
    img = original_img
    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-7].output
    final_conv_layer = get_output_layer(model, lay_name)
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output, class_weights])
    [conv_outputs, predictions, class_weights_output] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]
    class_weights_output = class_weights_output[0, :]
    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
    for i, w in enumerate(class_weights_output[:, ]):  # target_class
        cam += w * conv_outputs[:, :, i]
    print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + (original_img * 127.5 + 127.5) * 0.5
    img = np.uint8(img)
    cv2.imshow('show', img[0])
    cv2.waitKey(10000)


'''
To be able to create a CAM, the network architecture is restricted to have a global average pooling layer after the
 final convolutional layer, and then a linear (dense) layer. Unfortunately this means we can't apply this technique
 on existing networks that don't have this structure. What we can do is modify existing networks and fine tune them
 to get this. Designing network architectures to support tricks like CAM is like writing code in a way that makes
it easier to debug.

But I can see the whole vector(global avg) after the last conv layer, so we can use the 2048 and 7*7*2048 vector to see the heapmap.
'''
preprocessed_input = load_image(r'C:/Users/Ctbri/Desktop/110087_2_2_151819165.jpg', color_mode='grayscale')
# reverse the image for the pre model is trained like so
preprocessed_input = preprocessed_input[:, ::-1, :, :]
import util
import config

config.model = 'xception'
util.set_img_format()
model_module = util.get_model_class_instance()
model = model_module.load()
classes_in_keras_format = util.get_classes_in_keras_format()

predictions = model.predict(preprocessed_input)[
    0]  # predictions = model.predict([preprocessed_input,np.asarray([[1]])])[0]

predicted_class = np.argmax(predictions)
print(list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(predicted_class)])
visualize_class_activation_map(model, preprocessed_input, predicted_class, "block14_sepconv2_act")
