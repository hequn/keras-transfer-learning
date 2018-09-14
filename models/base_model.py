from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
from keras.utils import plot_model
import os

import config
import util


class BaseModel(object):
    def __init__(self,
                 class_weight=None,
                 nb_epoch=1000,
                 batch_size=32,
                 freeze_layers_number=None):
        self.model = None
        self.class_weight = class_weight
        self.nb_epoch = nb_epoch
        self.fine_tuning_patience = 20
        self.batch_size = batch_size
        self.freeze_layers_number = freeze_layers_number
        self.img_size = (224, 224)

    def _create(self):
        raise NotImplementedError('subclasses must override _create()')

    def _fine_tuning(self):
        self.freeze_top_layers()

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=1e-5),
            metrics=['accuracy'])
        self.model.summary()
        plot_model(self.model, to_file='model1.png', show_shapes=True)
        train_data = self.get_train_datagen(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                            preprocessing_function=self.preprocess_input)
        # early stopping
        callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(), patience=self.fine_tuning_patience)

        if util.is_keras2():
            self.model.fit_generator(
                train_data,
                steps_per_epoch=config.nb_train_samples / float(self.batch_size),
                epochs=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                validation_steps=config.nb_validation_samples / float(self.batch_size),
                callbacks=callbacks,
                class_weight=self.class_weight)
        else:
            self.model.fit_generator(
                train_data,
                samples_per_epoch=config.nb_train_samples,
                nb_epoch=self.nb_epoch,
                validation_data=self.get_validation_datagen(),
                nb_val_samples=config.nb_validation_samples,
                callbacks=callbacks,
                class_weight=self.class_weight)

        self.model.save(config.get_model_path())

    def train(self):
        print("Creating model...")
        self._create()
        # self.model.load_weights(config.get_fine_tuned_weights_path(), by_name=True, skip_mismatch=True)
        print("Model is created")
        print("Fine tuning...")
        self._fine_tuning()
        self.save_classes()
        print("Classes are saved")

    # add the default preprocess function in tf mode
    def preprocess_input(self, x):
        """Preprocesses a numpy array encoding a batch of images.

        # Arguments
            x: a 4D numpy array consists of RGB values within [0, 255].

        # Returns
            Preprocessed array.
        """
        return imagenet_utils.preprocess_input(x, mode='tf')

    def load(self):
        print("Creating model")
        self.load_classes()
        self._create()
        self.model.load_weights(config.get_fine_tuned_weights_path())
        return self.model

    @staticmethod
    def save_classes():
        joblib.dump(config.classes, config.get_classes_path())

    def get_input_tensor(self):
        if util.get_keras_backend_name() == 'theano':
            return Input(shape=(3,) + self.img_size)
        else:
            return Input(shape=self.img_size + (3,))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, patience=30, monitor='val_loss', embedding_mode=False,
                      target_object=None, visual_embedding=None, center_loss_input=False):
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        # if the embedding should be visualized
        if embedding_mode and visual_embedding is not None and target_object is not None:
            metadata_file_name = 'metadata.tsv'
            embeddings_metadata = {layer_name: metadata_file_name
                                   for layer_name in visual_embedding}
            test_data = util.write_csv_and_get_embedinput('.' + os.sep + 'sorted' + os.sep + 'test',
                                                          metadata_file_name,
                                                          '.' + os.sep + 'logs' + os.sep,
                                                          target_object,
                                                          limit_each_class=10,
                                                          center_loss_input=center_loss_input)
            tensor_board = TensorBoard(histogram_freq=0,
                                       # if there is histogram(not 0) we can not give the validation data in sequence
                                       write_graph=True,
                                       embeddings_freq=5,
                                       batch_size=32,
                                       write_images=True,
                                       embeddings_layer_names=visual_embedding,
                                       embeddings_metadata=embeddings_metadata,
                                       embeddings_data=test_data)

            return [early_stopping, model_checkpoint, tensor_board]
        else:
            return [early_stopping, model_checkpoint]

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

    @staticmethod
    def load_classes():
        config.classes = joblib.load(config.get_classes_path())

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return self.preprocess_input(x)[0]

    def get_train_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        # self.apply_mean(idg)
        return idg.flow_from_directory(config.train_dir, target_size=self.img_size, classes=config.classes)

    def get_validation_datagen(self, *args, **kwargs):
        idg = ImageDataGenerator(*args, **kwargs)
        # self.apply_mean(idg)
        return idg.flow_from_directory(config.validation_dir, target_size=self.img_size, classes=config.classes)

    # class TrainValTensorBoard(TensorBoard):
    #     def __init__(self, log_dir='./logs', **kwargs):
    #         # Make the original `TensorBoard` log to a subdirectory 'training'
    #         training_log_dir = os.path.join(log_dir, 'training')
    #         super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
    #
    #         # Log the validation metrics to a separate subdirectory
    #         self.val_log_dir = os.path.join(log_dir, 'validation')
    #
    #     def set_model(self, model):
    #         # Setup writer for validation metrics
    #         self.val_writer = tf.summary.FileWriter(self.val_log_dir)
    #         super(TrainValTensorBoard, self).set_model(model)
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         # Pop the validation logs and handle them separately with
    #         # `self.val_writer`. Also rename the keys so that they can
    #         # be plotted on the same figure with the training metrics
    #         logs = logs or {}
    #         val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
    #         for name, value in val_logs.items():
    #             summary = tf.Summary()
    #             summary_value = summary.value.add()
    #             summary_value.simple_value = value.item()
    #             summary_value.tag = name
    #             self.val_writer.add_summary(summary, epoch)
    #         self.val_writer.flush()
    #
    #         # Pass the remaining logs to `TensorBoard.on_epoch_end`
    #         logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    #         super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    #
    #     def on_train_end(self, logs=None):
    #         super(TrainValTensorBoard, self).on_train_end(logs)
    #         self.val_writer.close()


    # def generate_arrays_from_file(path):
    #     while 1:
    #         f = open(path)
    #         for line in f:
    #             # create numpy arrays of input data
    #             # and labels, from each line in the file
    #             x, y = process_line(line)
    #             img = load_images(x)
    #             yield (img, y)
    #         f.close()
    # model.fit_generator(generate_arrays_from_file('/my_file.txt'),
    #
    # samples_per_epoch=10000, nb_epoch=10)


    # datagen = ImageDataGenerator(
    #         featurewise_center=True, # set input mean to 0 over the dataset
    #         samplewise_center=False, # set each sample mean to 0
    #         featurewise_std_normalization=True, # divide inputs by std of the dataset
    #         samplewise_std_normalization=False, # divide each input by its std
    #         zca_whitening=False, # apply ZCA whitening
    #         rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
    #         width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
    #         height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
    #         horizontal_flip=True, # randomly flip images
    #         vertical_flip=False) # randomly flip images
    #
    # datagen.fit(X_sample) # let's say X_sample is a small-ish but statistically representative sample of your data
    #
    # # let's say you have an ImageNet generator that yields ~10k samples at a time.
    # for e in range(nb_epoch):
    #     print("epoch %d" % e)
    #     for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
    #         for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32): # these are chunks of 32 samples
    #             loss = model.train(X_batch, Y_batch)
    #
    # # Alternatively, without data augmentation / normalization:
    # for e in range(nb_epoch):
    #     print("epoch %d" % e)
    #     for X_train, Y_train in ImageNet(): # these are chunks of ~10k pictures
    #         model.fit(X_batch, Y_batch, batch_size=32, nb_epoch=1)
