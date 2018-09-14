import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

import config
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', help='Path to image', default=None, type=str)
    parser.add_argument('--accuracy', action='store_true', help='To print accuracy score')
    parser.add_argument('--plot_confusion_matrix', action='store_true')
    parser.add_argument('--execution_time', action='store_true')
    parser.add_argument('--store_activations', action='store_true')
    parser.add_argument('--novelty_detection', action='store_true')
    parser.add_argument('--model', type=str, required=True, help='Base model architecture',
                        choices=[config.MODEL_RESNET50, config.MODEL_RESNET152, config.MODEL_INCEPTION_V3,
                                 config.MODEL_VGG16])
    parser.add_argument('--data_dir', help='Path to data train directory')
    parser.add_argument('--batch_size', default=500, type=int, help='How many files to predict on at once')
    args = parser.parse_args()
    return args


def get_files(path):
    if os.path.isdir(path):
        files = glob.glob(path + '*.jpg')
    elif path.find('*') > 0:
        files = glob.glob(path)
    else:
        files = [path]

    if not len(files):
        print('No images found by the given path')
        exit(1)

    return files


def get_inputs_and_trues(files):
    inputs = []
    y_true = []

    for i in files:
        x = model_module.load_img(i)
        try:
            image_class = i.split(os.sep)[-2]
            keras_class = int(classes_in_keras_format[image_class])
            y_true.append(keras_class)
        except Exception:
            y_true.append(os.path.split(i)[1])

        inputs.append(x)

    return y_true, inputs


def predict(path):
    files = get_files(path)
    n_files = len(files)
    print('Found {} files'.format(n_files))

    if args.novelty_detection:
        activation_function = util.get_activation_function(model, model_module.noveltyDetectionLayerName)
        novelty_detection_clf = joblib.load(config.get_novelty_detection_model_path())

    y_trues = []
    predictions = np.zeros(shape=(n_files,))
    nb_batch = int(np.ceil(n_files / float(args.batch_size)))
    for n in range(0, nb_batch):
        print('Batch {}'.format(n))
        n_from = n * args.batch_size
        n_to = min(args.batch_size * (n + 1), n_files)

        y_true, inputs = get_inputs_and_trues(files[n_from:n_to])
        y_trues += y_true

        if args.store_activations:
            util.save_activations(model, inputs, files[n_from:n_to], model_module.noveltyDetectionLayerName, n)

        if args.novelty_detection:
            activations = util.get_activations(activation_function, [inputs[0]])
            nd_preds = novelty_detection_clf.predict(activations)[0]
            print(novelty_detection_clf.__classes[nd_preds])

        if not args.store_activations:
            # Warm up the model
            if n == 0:
                print('Warming up the model')
                start = time.clock()
                model.predict(np.array([inputs[0]]))
                end = time.clock()
                print('Warming up took {} s'.format(end - start))

            # Make predictions
            start = time.clock()
            ###############################
            # global out
            ###############################
            out = model.predict(np.array(inputs))
            print('out shape : ', out.shape)
            end = time.clock()
            predictions[n_from:n_to] = np.argmax(out, axis=1)
            print('Prediction on batch {} took: {}'.format(n, end - start))

    if not args.store_activations:
        #######################
        store_label = []
        ######################
        for i, p in enumerate(predictions):
            recognized_class = list(classes_in_keras_format.keys())[list(classes_in_keras_format.values()).index(p)]
            print('| should be {} ({}) -> predicted as {} ({}) -> score : {}'.format(y_trues[i],
                                                                                     files[i].split(os.sep)[-2], p,
                                                                                     recognized_class, np.max(out[i])))
            ###########################################
            if y_trues[i] == int(p):
                store_current_label = 1
            else:
                store_current_label = 0
            store_label.append([np.max(out[i]), store_current_label])
        # print(np.array(store_label).shape)
        # print(store_label)
        ###########################################
        if args.accuracy:
            print('Accuracy {}'.format(accuracy_score(y_true=y_trues, y_pred=predictions)))

        if args.plot_confusion_matrix:
            cnf_matrix = confusion_matrix(y_trues, predictions)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=False)
            util.plot_confusion_matrix(cnf_matrix, config.classes, normalize=True)
        #############
        return store_label


###################
# calc AP
###################
def metrix(data):
    import copy

    # data = [[0.23, 0], [0.76, 1], [0.01, 0], [0.91, 1], [0.13, 0],
    #         [0.45, 0], [0.12, 1], [0.03, 0], [0.38, 1], [0.11, 0],
    #         [0.03, 0], [0.09, 0], [0.65, 0], [0.07, 0], [0.12, 0],
    #         [0.24, 1], [0.1, 0], [0.23,  0], [0.46, 0], [0.08, 1]]


    # # list to numpy
    data = np.array(data)

    # according to the first column to sort
    # array([2, 7, 10, 13, 19, 11, 16, 9, 14, 6, 4, 0, 17, 15, 8, 5, 18, 12, 1, 3], dtype=int64)
    # means that 2 -> 0.01  7 -> 0.03  ....  3 -> 0.91
    data1 = np.lexsort(data[:, ::-1].T)
    # print(data[data1])

    # reverse order
    # array([ 3  1  8 15  6 19 12 18  5 17  0  4 14  9 16 11 13 10  7  2], dtype=int64)
    data1_reverse = data1[::-1]
    # print(data1_reverse)

    # sorted data
    # [[0.91 1] [0.76 1] ... [0.01 0]]
    data_result = data[data1_reverse]
    # print(data_result)

    # sorted label
    data_result_label = data_result[:, -1]
    # print(data_result_label)

    # calc true positive + false positive, that is those samples whose label is 1
    all_label_is_1_sum = sum(data_result_label == 1)
    # print(label_is_1_sum)

    precision = []
    recall = []

    # print(data_result_label[0:1])

    for i in range(1, len(data_result_label) + 1):
        # i starting from 1
        # while loop to the current
        current_label_is_1_sum = np.sum(data_result_label[:i] == 1)
        precision_temp = current_label_is_1_sum / len(data_result_label[:i])
        recall_temp = current_label_is_1_sum / all_label_is_1_sum
        precision.append(round(precision_temp, 10))
        recall.append(round(recall_temp, 10))

    # print('precision:',  precision)
    # print('recall:', recall)

    assert len(recall) == len(precision) and len(recall) > 0

    temp = []
    for i in range(len(recall)):
        current_recall_value_sum = recall.count(recall[i])
        if current_recall_value_sum > 1:
            # current recall index  first
            current_recall_index = recall.index(recall[i])
            current_recall_info = [current_recall_index, current_recall_value_sum]
            if current_recall_info not in temp:
                temp.append(current_recall_info)

    # print(temp)

    max_precision = copy.deepcopy(precision)

    for temp_every in temp:
        precision_start_index = temp_every[0]
        precision_end_index = temp_every[1] + temp_every[0]  # not include

        # get max
        temp_max = np.max(max_precision[temp_every[0]:(temp_every[1] + temp_every[0])])

        for i in range(precision_start_index, precision_end_index):
            max_precision[i] = temp_max

    # print('max_presicion : ', max_precision)


    total = sum(np.array(max_precision))
    print(total / len(max_precision))
    return (total / len(max_precision))


if __name__ == '__main__':
    tic = time.clock()

    args = parse_args()
    print('=' * 50)
    print('Called with args:')
    print(args)

    if args.data_dir:
        config.data_dir = args.data_dir
        config.set_paths()
    if args.model:
        config.model = args.model

    util.set_img_format()
    model_module = util.get_model_class_instance()
    model = model_module.load()

    classes_in_keras_format = util.get_classes_in_keras_format()

    all_metrix = []
    print('args.path:', os.listdir(args.path))
    for cow_dir in os.listdir(args.path):
        root = args.path + '/' + str(cow_dir) + '/'
        store_label = predict(root)
        # print(store_label)
        all_metrix.append(metrix(store_label))

    total = 0
    for i in range(len(all_metrix)):
        total += all_metrix[i]
    print('mAP:', (total / len(all_metrix)))

    if args.execution_time:
        toc = time.clock()
        print('Time: %s' % (toc - tic))
