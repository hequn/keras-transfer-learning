from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.preprocessing import image
import numpy as np
import config
from .base_model import BaseModel


class InceptionV3(BaseModel):
    noveltyDetectionLayerName = 'fc3'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        super(InceptionV3, self).__init__(*args, **kwargs)

        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80

        self.img_size = (224, 224)

    def _create(self):
        base_model = KerasInceptionV3(weights='imagenet', include_top=False, input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(base_model)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)#BatchNormalization()(x)
        x = Dense(9192, activation='elu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(9192, activation='elu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        predictions = Dense(len(config.classes), activation='softmax')(x)

        self.model = Model(input=base_model.input, output=predictions)

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
        pass

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),  # SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=['accuracy'])

        self.model.fit_generator(
            self.get_train_datagen(rotation_range=15.,
                                   horizontal_flip=False,
                                   # rescale=1./255,
                                   width_shift_range=0.05, height_shift_range=0.05, channel_shift_range=5,
                                   shear_range=0.05, zoom_range=0.1,
                                   preprocessing_function=self.preprocess_input),
            samples_per_epoch=config.nb_train_samples,
            nb_epoch=self.nb_epoch,
            validation_data=self.get_validation_datagen(preprocessing_function=self.preprocess_input),
            nb_val_samples=config.nb_validation_samples,
            callbacks=self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience),
            class_weight=self.class_weight)

        self.model.save(config.get_model_path())


def inst_class(*args, **kwargs):
    return InceptionV3(*args, **kwargs)
