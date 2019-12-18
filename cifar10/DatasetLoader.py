import pickle
import numpy as np


def readMetaData(datasetDir):
    with open(datasetDir + "/batches.meta", "rb") as f:
        res = pickle.load(f, encoding='bytes')
    return res[b'label_names'], res[b'num_cases_per_batch'], res[b'num_vis']


def readTrainDataset(datasetDir):
    x_train = np.array([]).reshape(0, 3072)
    y_train = []
    file_names = []

    for i in range(1, 6):
        with open(datasetDir + "/data_batch_" + str(i), "rb") as f:
            res = pickle.load(f, encoding='bytes')
            x_train = np.concatenate((x_train, res[b'data']))
            y_train += res[b'labels']
            file_names += res[b'filenames']
    file_names = [x.decode('utf-8') for x in file_names]
    return x_train, np.array(y_train), file_names


def readTestDataset(datasetDir):
    x_test = np.array([]).reshape(0, 3072)
    y_test = []
    file_names = []

    with open(datasetDir + "/test_batch", "rb") as f:
        res = pickle.load(f, encoding='bytes')
        x_test = np.concatenate((x_test, res[b'data']))
        y_test += res[b'labels']
        file_names += res[b'filenames']

    file_names = [x.decode('utf-8') for x in file_names]
    return x_test, np.array(y_test), file_names