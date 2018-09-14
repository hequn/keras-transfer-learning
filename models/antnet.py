import keras
import numpy as np
from keras import backend as K
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Conv2D, SeparableConv2D, Lambda, Embedding, \
    Activation, MaxPooling2D, add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

import config
import util
from tools.triplet_loss import batch_hard_triplet_loss
from .base_model import BaseModel


class AntNet(BaseModel):
    """E.G. The Xception model illustrate the fine tuned procedure which I give three loss mode choices here."""
    noveltyDetectionLayerName = 'embedding'
    noveltyDetectionLayerSize = 3
    # 0 - >triplet_loss, 1 -> center_loss, 2 -> normal default
    loss_choice = 1

    def __init__(self, *args, **kwargs):
        super(AntNet, self).__init__(*args, **kwargs)

        # if not self.freeze_layers_number:
        # which blocks to be trained, the layers will be frozen
        # self.freeze_layers_number = 80
        # img_size, of course
        self.img_size = (224, 224)

    def _create(self):
        img_input = self.get_input_tensor()

        '''START'''
        x = Conv2D(32, (3, 3),
                   strides=(2, 2),
                   use_bias=False,
                   name='block1_conv1')(img_input)
        x = BatchNormalization(name='block1_conv1_bn')(x)
        x = Activation('relu', name='block1_conv1_act')(x)
        x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
        x = BatchNormalization(name='block1_conv2_bn')(x)
        x = Activation('relu', name='block1_conv2_act')(x)

        residual = Conv2D(128, (1, 1),
                          strides=(2, 2),
                          padding='same',
                          use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block2_sepconv1')(x)
        x = BatchNormalization(name='block2_sepconv1_bn')(x)
        x = Activation('relu', name='block2_sepconv2_act')(x)
        x = SeparableConv2D(128, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block2_sepconv2')(x)
        x = BatchNormalization(name='block2_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3),
                         strides=(2, 2),
                         padding='same',
                         name='block2_pool')(x)
        x = add([x, residual])

        residual = Conv2D(256, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block3_sepconv1_act')(x)
        x = SeparableConv2D(256, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block3_sepconv1')(x)
        x = BatchNormalization(name='block3_sepconv1_bn')(x)
        x = Activation('relu', name='block3_sepconv2_act')(x)
        x = SeparableConv2D(256, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block3_sepconv2')(x)
        x = BatchNormalization(name='block3_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2),
                         padding='same',
                         name='block3_pool')(x)
        x = add([x, residual])

        residual = Conv2D(728, (1, 1),
                          strides=(2, 2),
                          padding='same',
                          use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu', name='block4_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block4_sepconv1')(x)
        x = BatchNormalization(name='block4_sepconv1_bn')(x)
        x = Activation('relu', name='block4_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3),
                            padding='same',
                            use_bias=False,
                            name='block4_sepconv2')(x)
        x = BatchNormalization(name='block4_sepconv2_bn')(x)

        x = MaxPooling2D((3, 3), strides=(2, 2),
                         padding='same',
                         name='block4_pool')(x)
        x = add([x, residual])
        '''END'''

        base_model = Model(img_input, x, name='antnet')
        # self.make_net_layers_non_trainable(base_model)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)#BatchNormalization()(x)
        x = Dense(24, activation='elu', name='fc1')(x)
        x = Dropout(0.5)(x)
        # record the embedding layer in order to get the vector represents the target identity
        embedding = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax', name='classify_dense')(embedding)
        # triplet loss mode, center loss mode, default loss mode
        if self.loss_choice == 0:
            self.model = Model(inputs=[base_model.input], outputs=[predictions, embedding])
        elif self.loss_choice == 1:
            input_target = Input(shape=(1,))  # single value ground truth labels as inputs
            centers = Embedding(len(config.classes), self.noveltyDetectionLayerSize)(input_target)
            # make the Lambda Layer and let it to be the last output
            l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True),
                             name='l2_loss')([embedding, centers])
            self.model = Model(inputs=[base_model.input, input_target], outputs=[predictions, l2_loss])
        elif self.loss_choice == 2:
            self.model = Model(input=base_model.input, output=predictions)

    '''Normalize the input data to [-1,1]'''

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return self.preprocess_input(x)[0]

    @staticmethod
    def apply_mean(image_data_generator):
        # if we should use z-white or sample-wise center-wise center,
        # we must use image_data_generator.fit(data) to calculate the mean and std.
        # the data is better to be some represented samples in numpy array but not all data.
        pass

    def triplet_loss_fn(self, y_true, y_pred):
        # Convert the float to int value
        y_true = K.tf.cast(y_true, K.tf.int64)
        # Get the label from the first row in order to give the class index in shape (batch,)
        y_true = y_true[:, 0]
        return batch_hard_triplet_loss(y_true, y_pred, 0.5, squared=False)

    def center_loss_fn(self, y_true, y_pred):
        # l2_loss layer output is the y_pred, the value will be passed to the optimizer then
        return y_pred

    def _fine_tuning(self):
        # self.model.load_weights(config.get_fine_tuned_weights_path())
        self.freeze_top_layers()
        # let's decide the mode and set the matched compile method and data generator
        # the under codes may seemed to be repeated, but I prefer to make it easier to understand and read.
        if self.loss_choice == 0:
            self.model.compile(
                loss=['categorical_crossentropy', self.triplet_loss_fn],
                optimizer=Adam(lr=1e-5),
                loss_weights=[1, 1],
                # SGD(lr=1e-5, decay=1e-6, momentum=0.8, nesterov=True),#Adadelta(),#Adam(lr=1e-5),
                metrics=['accuracy'])
            train_data_generator = util.triplet_transformed_generator(self.get_train_datagen(rotation_range=10.,
                                                                                             shear_range=0.05,
                                                                                             zoom_range=0.1,
                                                                                             width_shift_range=0.05,
                                                                                             height_shift_range=0.05,
                                                                                             horizontal_flip=True,
                                                                                             preprocessing_function=self.preprocess_input),
                                                                      self.noveltyDetectionLayerSize)
            validation_data_generator = util.triplet_transformed_generator(
                self.get_validation_datagen(preprocessing_function=self.preprocess_input),
                self.noveltyDetectionLayerSize)
        elif self.loss_choice == 1:
            self.model.compile(
                loss=["categorical_crossentropy", self.center_loss_fn],
                optimizer=Adam(lr=1e-5),
                loss_weights=[1, 0.15],
                metrics=['accuracy'])
            train_data_generator = util.centerloss_transformed_generator(self.get_train_datagen(rotation_range=10.,
                                                                                                shear_range=0.05,
                                                                                                zoom_range=0.1,
                                                                                                width_shift_range=0.05,
                                                                                                height_shift_range=0.05,
                                                                                                horizontal_flip=True,
                                                                                                preprocessing_function=self.preprocess_input))
            validation_data_generator = util.centerloss_transformed_generator(
                self.get_validation_datagen(preprocessing_function=self.preprocess_input))
        elif self.loss_choice == 2:
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr=1e-5),
                metrics=['accuracy'])
            train_data_generator = self.get_train_datagen(rotation_range=10.,
                                                          shear_range=0.05,
                                                          zoom_range=0.1,
                                                          width_shift_range=0.05,
                                                          height_shift_range=0.05,
                                                          horizontal_flip=True,
                                                          preprocessing_function=self.preprocess_input)
            validation_data_generator = self.get_validation_datagen(preprocessing_function=self.preprocess_input)

        # print the mode construction and key info
        self.model.summary()
        # fit the generator in batch sequence mode
        self.model.fit_generator(
            train_data_generator,
            steps_per_epoch=config.nb_train_samples / self.batch_size,
            epochs=self.nb_epoch,
            validation_data=validation_data_generator,
            validation_steps=config.nb_validation_samples / self.batch_size,
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(),
                                         patience=self.fine_tuning_patience,
                                         embedding_mode=(True if self.loss_choice is not 2 else False),
                                         target_object=self,
                                         visual_embedding=[self.noveltyDetectionLayerName],
                                         center_loss_input=(True if self.loss_choice is 1 else False)),
            class_weight=self.class_weight)
        #self.model.save(config.get_model_path())
        keras.models.save_model(self.model, config.get_model_path())

def inst_class(*args, **kwargs):
    return AntNet(*args, **kwargs)
