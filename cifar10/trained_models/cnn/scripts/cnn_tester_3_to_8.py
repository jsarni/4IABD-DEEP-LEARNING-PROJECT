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

    ##Scirpt du 26 01 2020 23h23
    # ######################## CNN 3-8 N N O N 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_NNON_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc
    # ######################## CNN 3-8 O O O O 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=True, use_l1l2_conv=True, use_l1l2_output=True,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_OOOO_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc
    # ######################## CNN 3-8 O O N N 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=True, use_l1l2_conv=False, use_l1l2_output=False,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_OONN_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc
    # ######################## CNN 3-8 O N O N 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_ONON_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc
    # ######################## CNN 3-8 N N O N 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_NNON_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc
    # ######################## CNN 3-8 O N O O 32 64 #########################
    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=True,
                                       min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=64)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_3_8_ONNO_32_64', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc