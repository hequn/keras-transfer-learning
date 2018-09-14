from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model,load_model
from keras.layers import Flatten, Dense, Dropout, Lambda, Dot, dot
from keras import backend as K

import config
from .base_model import BaseModel


class VGG16(BaseModel):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):
        super(VGG16, self).__init__(*args, **kwargs)

    def _create(self):
        # base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        # self.make_net_layers_non_trainable(base_model)
        #
        # test1 = base_model.get_layer('block5_conv3').output
        # test1 = Lambda(lambda x: K.permute_dimensions(x, (0, 3, 1, 2)))(test1) #K.permute_dimensions(test1,(0, 3, 1, 2))  # shift to [Batch, Channel, Height, Width] ==> [B,512,28,28]
        # test1 = Lambda(lambda x: K.reshape(x, [-1, 512, 14*14]))(test1)#K.reshape(test1, [-1, 512, 28 * 28])
        # #test1_T = Lambda(lambda x: K.permute_dimensions(test1,[0,2,1]))(test1)
        #
        # x_value = Lambda(lambda x: K.batch_dot(x, K.permute_dimensions(x,[0,2,1])))(test1)
        # x_value = Lambda(lambda x: K.reshape(x, [-1, 512*512]))(x_value) # K.reshape(x_value, [-1, 512 * 512])
        # y_value = Lambda(lambda x: K.sqrt(x + 1e-10))(x_value)  #K.sqrt(x_value + 1e-10)
        # z_value = Lambda(lambda x: K.l2_normalize(x, axis=1))(y_value)#K.l2_normalize(y_value, axis=1)
        #
        # # x = base_model.output
        # # z_value = Flatten()(z_value)
        # x = Dense(128, activation='elu', name='fc1')(z_value)
        # x = Dropout(0.6)(x)
        # x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        # x = Dropout(0.6)(x)
        # predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)
        # self.model = Model(input=base_model.input, output=predictions)
        #self.model.load_weights()
        base_model = KerasVGG16(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        print("the vgg16 layers is!!!!!!!!!!!!!!!!!!!!!!!")
        print(base_model.layers[3].output.shape)
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=base_model.input, output=predictions)
        self.model = load_model('C:/Users/Ctbri/Desktop/chenxingli/keras-transfer-learning-for-oxford102/trained/model-vgg16.h5')
def inst_class(*args, **kwargs):
    return VGG16(*args, **kwargs)
