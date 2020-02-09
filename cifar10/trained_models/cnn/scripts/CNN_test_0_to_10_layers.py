from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
import tensorflow as tf

from cifar10.models.CNN import *
from cifar10.models.ModelTester import test_models

if __name__ == "__main__":
    #déclaration des param:
    epochs=[30]
    #importation des données:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    #définition des noms des classes:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    epochs=[50]
    for i in range(3):
        clear_session()
        cnn_str=generateRandoCNNStruc(use_maxpool=False,use_l1l2_conv=False,use_l1l2_output=True,use_dropout=False,min_nb_layers=3,max_nb_layers=4,min_filter_size=32,max_filter_size=128)
        cnn_model=[create_CNN_model(cnn_str)]
        cnn_desc=[getcnnStructAsString(cnn_str)]
        test_models('cnn_3_4_N_N_O_N_32_128', cnn_model, cnn_desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=1024, save_image=True)
        del cnn_model
        del cnn_desc