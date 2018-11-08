import matplotlib;

matplotlib.use('Agg')  # fixes issue if no GUI provided
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import importlib
import keras
from keras import backend as K
import config
import math
import itertools


def save_history(history, prefix):
    if 'acc' not in history.history:
        return

    if not os.path.exists(config.plots_dir):
        os.mkdir(config.plots_dir)

    img_path = os.path.join(config.plots_dir, '{}-%s.jpg'.format(prefix))

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(img_path % 'accuracy')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(img_path % 'loss')
    plt.close()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    confusion_matrix_dir = './confusion_matrix_plots'
    if not os.path.exists(confusion_matrix_dir):
        os.mkdir(confusion_matrix_dir)

    plt.cla()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="#BFD1D4" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if normalize:
        plt.savefig(os.path.join(confusion_matrix_dir, 'normalized.jpg'))
    else:
        plt.savefig(os.path.join(confusion_matrix_dir, 'without_normalization.jpg'))


def get_dir_imgs_number(dir_path):
    allowed_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    number = 0
    for e in allowed_extensions:
        number += len(glob.glob(os.path.join(dir_path, e)))
    return number


def set_samples_info():
    """Walks through the train and valid directories
    and returns number of images"""
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    dirs_info = {config.train_dir: 0, config.validation_dir: 0}
    for d in dirs_info:
        iglob_iter = glob.iglob(d + '**/*.*')
        for i in iglob_iter:
            filename, file_extension = os.path.splitext(i)
            if file_extension[1:] in white_list_formats:
                dirs_info[d] += 1

    config.nb_train_samples = dirs_info[config.train_dir]
    config.nb_validation_samples = dirs_info[config.validation_dir]


def get_class_weight(d):
    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
    class_number = dict()
    dirs = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])
    k = 0
    for class_name in dirs:
        class_number[k] = 0
        iglob_iter = glob.iglob(os.path.join(d, class_name, '*.*'))
        for i in iglob_iter:
            _, ext = os.path.splitext(i)
            if ext[1:] in white_list_formats:
                class_number[k] += 1
        k += 1

    total = np.sum(list(class_number.values()))
    max_samples = np.max(list(class_number.values()))
    mu = 1. / (total / float(max_samples))
    keys = class_number.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu * total / float(class_number[key]))
        class_weight[key] = score if score > 1. else 1.

    return class_weight


def set_classes_from_train_dir():
    """Returns classes based on directories in train directory"""
    d = config.train_dir
    config.classes = sorted([o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))])


def override_keras_directory_iterator_next():
    """Overrides .next method of DirectoryIterator in Keras
      to reorder color channels for images from RGB to BGR"""
    from keras.preprocessing.image import DirectoryIterator

    original_next = DirectoryIterator.next

    # do not allow to override one more time
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)
        # we must be sure the channel first or channel last to reorder
        # for tensorflow it is the channel last
        batch_x = batch_x[:, :, :, ::-1]
        return batch_x, batch_y
        # If you want to visualize the images to see the channel flip
        # batch_x = np.array((batch_x / 2. + 0.5) * 255, dtype=np.int32)
        # plt.subplot(121)
        # plt.imshow(batch_x[0])
        # plt.title("Original1")
        # batch_x = batch_x[:, ::, :, ::-1]
        # plt.subplot(122)
        # plt.imshow(batch_x[0])
        # plt.title("Original2")
        # plt.show()

    DirectoryIterator.next = custom_next


def get_classes_in_keras_format():
    if config.classes:
        return dict(zip(config.classes, range(len(config.classes))))
    return None


def get_model_class_instance(*args, **kwargs):
    module = importlib.import_module("models.{}".format(config.model))
    return module.inst_class(*args, **kwargs)


def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(activation_function, X_batch):
    activations = activation_function([X_batch, 0])
    return activations[0][0]


def save_activations(model, inputs, files, layer, batch_number):
    all_activations = []
    ids = []
    af = get_activation_function(model, layer)
    for i in range(len(inputs)):
        acts = get_activations(af, [inputs[i]])
        all_activations.append(acts)
        ids.append(files[i].split('\\')[-2])

    submission = pd.DataFrame(all_activations)
    submission.insert(0, 'class', ids)
    submission.reset_index()
    if batch_number > 0:
        submission.to_csv(config.activations_path, index=False, mode='a', header=False)
    else:
        submission.to_csv(config.activations_path, index=False)


def lock():
    if os.path.exists(config.lock_file):
        exit('Previous process is not yet finished.')

    with open(config.lock_file, 'w') as lock_file:
        lock_file.write(str(os.getpid()))


def unlock():
    if os.path.exists(config.lock_file):
        os.remove(config.lock_file)


def is_keras2():
    return keras.__version__.startswith('2')


def get_keras_backend_name():
    try:
        return K.backend()
    except AttributeError:
        return K._BACKEND


def tf_allow_growth():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    set_session(sess)


def set_img_format():
    try:
        if K.backend() == 'theano':
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')
    except AttributeError:
        if K._BACKEND == 'theano':
            K.set_image_dim_ordering('th')
        else:
            K.set_image_dim_ordering('tf')


def triplet_transformed_generator(generator, embedding_size):
    """The data generator drive for the triplet loss model specially."""
    while True:
        # get the transformed batch images
        data = next(generator)
        # reformatted the output shapes to fit the losses for triplet
        x = [data[0]]
        # get the class index from the categorical formatted np array
        batch_class_index = data[1].argmax(axis=-1).reshape(data[1].shape[0], 1)
        # padding zeros to the extra columns
        padding_batch_class_index = np.zeros([data[1].shape[0], embedding_size])
        padding_batch_class_index[:batch_class_index.shape[0], :batch_class_index.shape[1]] = batch_class_index
        # for the y[1](y_true) shape should be same with the y_pred, the above things have to be done
        y = [data[1], padding_batch_class_index]
        yield x, y


def centerloss_transformed_generator(generator):
    """The data generator drive for the centor loss model specially."""
    while True:
        # get the transformed batch images
        data = next(generator)
        # get the class index from the categorical formatted np array
        batch_class_index = data[1].argmax(axis=-1).reshape(data[1].shape[0],)
        # reformatted the output shapes to fit the losses for centor loss
        x = [data[0], batch_class_index]
        y = [data[1], np.random.rand(data[1].shape[0],)]
        yield x, y


def write_csv_and_get_embedinput(data_path, csv_name, folder_path, model_module, limit_each_class=10, center_loss_input=False):
    """In order to use the Tensorboard in Keras callbacks, the test data and tsv file should be prepared"""

    def get_files(path):
        """get the jpg files under the path"""
        if os.path.isdir(path):
            files = glob.glob(path + '*.jpg')
        elif path.find('*') > 0:
            files = glob.glob(path)
        else:
            files = [path]
        if not len(files):
            print('No images found by the given path')
            return []
        return files[0:limit_each_class]

    def get_inputs_and_trues(files, model_module):
        """get the label and the transformed(preprocessed) images"""
        inputs = []
        y_true = []
        for i in files:
            # the load img the actual model instance
            x = model_module.load_img(i)
            try:
                image_class = i.split(os.sep)[-2]
                keras_class = int(get_classes_in_keras_format()[image_class])
                y_true.append(keras_class)
            except Exception:
                y_true.append(os.path.split(i)[1])
            inputs.append(x)
        return y_true, inputs

    labels_csv = []
    inputs_all = []

    # find the labels and inputs, arrays with numpy arrays
    for dir in os.listdir(data_path):
        root = data_path + os.sep + str(dir) + os.sep
        y_true, inputs = get_inputs_and_trues(get_files(root), model_module)
        if len(y_true) != 0:
            labels_csv.append(y_true)
            inputs_all.append(inputs)
    # reshape the label array, flat it
    labels_csv = np.array(labels_csv, dtype=np.int32).reshape(-1)
    inputs_all = np.array(inputs_all, dtype=np.float32)
    if len(inputs_all) == 0:
        raise FileNotFoundError('Please make true the test files are placed well or the path is wrong.')
    # downgrade the dimension , multiply the [0] and [1], keep the HWC
    inputs_all = inputs_all.reshape(inputs_all.shape[0] * inputs_all.shape[1],
                                    inputs_all.shape[2], inputs_all.shape[3], inputs_all.shape[4])
    metadata_filename = csv_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # write to csv file
    with open(folder_path + metadata_filename, 'w') as f:
        for i in range(len(labels_csv)):
            f.write('{}\n'.format(labels_csv[i]))

    if not center_loss_input:
        return inputs_all
    elif center_loss_input:
        return [inputs_all, labels_csv]
