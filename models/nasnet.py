from keras.applications.nasnet import NASNetLarge as KerasNASNet
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.preprocessing import image
import numpy as np
import os
from  keras.utils import  plot_model

import config
from .base_model import BaseModel


class NASNet(BaseModel):
    noveltyDetectionLayerName = 'fc3'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(NASNet, self).__init__(*args, **kwargs)

        if not self.freeze_layers_number:
            self.freeze_layers_number = 122

        self.img_size = (224, 224)

    def _create(self):
        base_model = KerasNASNet(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='elu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='elu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax')(x)

        self.model = Model(input=base_model.input, output=predictions)

    def preprocess_input(self, x):
        print(x)
        print(x.shape)
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
        pass

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='model1.png', show_shapes=True)
        self.model.fit_generator(
            self.get_train_datagen(rotation_range=15.,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   horizontal_flip=True,
                                   preprocessing_function=self.preprocess_input),
            samples_per_epoch=config.nb_train_samples,
            nb_epoch=self.nb_epoch,
            validation_data=self.get_validation_datagen(preprocessing_function=self.preprocess_input),
            nb_val_samples=config.nb_validation_samples,
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(),os.path.join(os.path.abspath("."),"checkpoint"),patience=self.fine_tuning_patience),
            class_weight=self.class_weight)

        self.model.save(config.get_model_path())


def inst_class(*args, **kwargs):
    return NASNet(*args, **kwargs)
