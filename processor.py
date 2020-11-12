from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
from os import listdir
from os.path import join
import logging
import sys

MODE_SPLITTED = 0
MODE_MIXED = 1

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class Processor(object):
    def __init__(self, mode=MODE_MIXED, dataset='tnews', lang='cn'):
        self.mode = mode
        self.dataset = dataset
        self.lang = lang

    def get_data(self, num_classes):
        files = [f for f in listdir(join('../data', self.dataset))
                 if str(num_classes) in f and 'features' not in f and '.pkl' in f]

        if len(files) != 1:
            # files = [files[1]]
            print(files)
            print('not single pkl file', files)
            sys.exit()

        pkl_path = join('../data', self.dataset, files[0])
        logging.info('pkl_path: {}'.format(pkl_path))
        with open(pkl_path, 'rb') as f:
            dic = pickle.load(f)

        dev_x, dev_y = None, None
        if self.mode == MODE_SPLITTED:
            train_x = dic['train_x']
            train_y = dic['train_y']
            test_x = dic['test_x']
            test_y = dic['test_y']
            if 'dev_x' in dic:
                dev_x = dic['dev_x']
                dev_y = dic['dev_y']
            y = train_y + test_y
            if dev_y is not None:
                y += dev_y
        elif self.mode == MODE_MIXED:
            if 'x' not in dic.keys():
                train_x, train_y, test_x, test_y = dic['train_x'], dic['train_y'], dic['test_x'], dic['test_y']
                y = train_y + test_y
            else:
                x = dic['x']
                y = dic['y']
                test_size = int(len(y) * 0.2)
                test_x, test_y = x[:test_size], y[:test_size]
                train_x, train_y = x[test_size:], y[test_size:]

        label_encoder = LabelEncoder()  # [0, num_classes-1]
        label_encoder.fit(y)

        label_encoded = label_encoder.transform(train_y)
        train_y = to_categorical(label_encoded)  # categorical
        label_encoded = label_encoder.transform(test_y)
        test_y = to_categorical(label_encoded)  # categorical

        if dev_y is not None:
            label_encoded = label_encoder.transform(dev_y)
            dev_y = to_categorical(label_encoded)  # categorical

        num_classes = train_y.shape[1]
        return train_x, train_y, test_x, test_y, dev_x, dev_y, num_classes, label_encoder


class ProcessorFeature(object):
    def __init__(self, mode=MODE_MIXED, dataset='tnews', lang='cn'):
        self.mode = mode
        self.dataset = dataset
        self.lang = lang

    def get_data(self, num_classes):
        files = [f for f in listdir(join('../data', self.dataset))
                 if 'features.fixed.pkl' in f
                 and str(num_classes) in f]

        if len(files) != 1:
            print('not single pkl file', files)
            sys.exit()

        pkl_path = join('../data', self.dataset, files[0])
        logging.info('pkl_path: {}'.format(pkl_path))
        with open(pkl_path, 'rb') as f:
            dic = pickle.load(f)

        dev_x, dev_x_len, dev_y = None, None, None

        train_x = dic['train_x']
        train_x_len = dic['train_x_len']
        train_y = dic['train_y']
        test_x = dic['test_x']
        test_x_len = dic['test_x_len']
        test_y = dic['test_y']
        if 'dev_x' in dic:
            dev_x = dic['dev_x']
            dev_x_len = dic['dev_x_len']
            dev_y = dic['dev_y']

        num_classes = train_y.shape[1]
        return train_x, train_x_len, train_y, test_x, test_x_len, test_y, \
               dev_x, dev_x_len, dev_y, num_classes
