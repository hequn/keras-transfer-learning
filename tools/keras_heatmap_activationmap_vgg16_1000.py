import argparse
import glob

import cv2
from keras.callbacks import *
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Dense, Lambda
from keras.models import *
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical


def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "/neg")
    pos_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(pos_path + "/*.png")]
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [cv2.resize(cv2.imread(x), (64, 128)) for x in glob.glob(neg_path + "/*.png")]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)

    return X, y

def train(dataset_path):
    model = get_model()
    X, y = load_inria_person(dataset_path)
    print
    "Training.."
    checkpoint_path = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False,
                                 save_weights_only=False, mode='auto')
    model.fit(X, y, nb_epoch=40, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])


def visualize_class_activation_map(model_path, img_path, output_path):
    model = load_model(model_path)
    original_img = cv2.imread(img_path, 1)
    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "conv5_3")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
    for i, w in enumerate(class_weights[:, 1]):
        cam += w * conv_outputs[i, :, :]
    print("predictions", predictions)
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap[np.where(cam < 0.2)] = 0
    img = heatmap * 0.5 + original_img
    cv2.imwrite(output_path, img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=False, help='Train the network or visualize a CAM')
    parser.add_argument("--image_path", type=str, help="Path of an image to run the network on")
    parser.add_argument("--output_path", type=str, default="heatmap.jpg", help="Path of an image to run the network on")
    parser.add_argument("--model_path", type=str, help="Path of the trained model")
    parser.add_argument("--dataset_path", type=str, help= \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    args = parser.parse_args()
    return args


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def VGG16_convolutions():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, None, None)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    return model


def get_model():
    model = VGG16_convolutions()

    model = load_model_weights(model, "vgg16_weights.h5")

    model.add(Lambda(global_average_pooling,
                     output_shape=global_average_pooling_shape))
    model.add(Dense(2, activation='softmax', init='uniform'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def load_model_weights(model, weights_path):
    print('Loading model.')
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False
    f.close()
    print
    'Model loaded.'
    return model


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

if __name__ == '__main__':
    args = get_args()
    if args.train:
        train(args.dataset_path)
    else:
        visualize_class_activation_map(args.model_path, args.image_path, args.output_path)