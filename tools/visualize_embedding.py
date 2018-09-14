import time
import os
import cv2
import config
import util
import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from keras.applications.imagenet_utils import preprocess_input


def predict():
    test_data = util.write_csv_and_get_embedinput('.' + os.sep + 'sorted' + os.sep + 'test',
                                                  'metadata.tsv',
                                                  '.' + os.sep + 'logs' + os.sep,
                                                  model_module,
                                                  limit_each_class=10,
                                                  center_loss_input=False)
    print('Warming up the model')
    start = time.clock()
    input_shape = (1,) + model_module.img_size + (3,)
    dummpy_img = np.ones(input_shape)
    dummpy_img = preprocess_input(dummpy_img)
    model.predict(dummpy_img)
    end = time.clock()
    print('Warming up took {} s'.format(end - start))

    batch_size = 32
    batch_num = int(len(test_data) / batch_size + 1)
    out_all = np.array([], dtype=np.float32).reshape(-1, 1024)
    for i in range(batch_num):
        n_from = i * batch_size
        n_to = min((i + 1) * batch_size, len(test_data))
        # Make predictions
        start = time.clock()
        out = model.predict(test_data[n_from:n_to])
        out_all = np.append(out_all, out[1], axis=0)
        print('Prediction batch {} took: {}, total epoch is {}'.format(i + 1, end - start, batch_num))
    return out_all, test_data


# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding

    Args:
      data: NxHxW[x3] tensor containing the images.

    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    # data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


if __name__ == '__main__':
    tic = time.clock()

    config.model = 'xception'

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    # EMBEDDINGS VISUALIZATION

    # Compute embeddings on the test set
    tf.logging.info("Predicting")
    predictions, img_data = predict()  # args.path

    tf.reset_default_graph()

    tf.logging.info("Embeddings shape: {}".format(predictions.shape))

    sprite = images_to_sprite(img_data)
    cv2.imwrite(os.path.join('./logs', 'sprite_classes.png'), sprite)

    # Visualize test embeddings
    embedding_var = tf.Variable(predictions, name='embeddings')

    eval_dir = os.path.join('./logs', "")
    summary_writer = tf.summary.FileWriter(eval_dir)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.sprite.image_path = pathlib.Path(os.path.join('./logs', 'sprite_classes.png')).name
    embedding.sprite.single_image_dim.extend([224, 224])

    # Specify where you find the metadata
    # Save the metadata file needed for Tensorboard projector
    metadata_filename = "metadata.tsv"
    embedding.metadata_path = metadata_filename

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(embedding_var.initializer)
        saver.save(sess, os.path.join(eval_dir, "embeddings.ckpt"))
