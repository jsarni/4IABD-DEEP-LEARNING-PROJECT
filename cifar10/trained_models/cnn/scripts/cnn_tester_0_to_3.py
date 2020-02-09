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

    #script du 06/02/2020:
    for e in (30,35):
        (modelcnn, desccnn) = generateCNNModels([3], [
            [(128, 3),(64, 3),(32, 3)]],
                                                ['relu'], ['softmax'], ['same'], [True], [[2]], [3], [True], [[0]],
                                                [0.25], [True], [True], [0.030], [0.090], [[1,3]],
                                                ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_1_3_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                    epochs_p=[e],
                    batch_size_p=256, save_image=True)
        clear_session()



    # ##Scirpt du 26 01 2020 11h40:
    # # ######################## CNN 1-3 N N N N 32 64 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=False, use_l1l2_conv=False, use_l1l2_output=False,
    #                                    min_nb_layers=1, max_nb_layers=3, min_filter_size=32, max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_3_NNNN_32_64', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    # # ######################## CNN 1-3 O N N N 32 64 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=False, use_l1l2_output=False,
    #                                    min_nb_layers=1, max_nb_layers=3, min_filter_size=32, max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_3_ONNN_32_64', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    # # ######################## CNN 1-3 O N O N 32 64 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=1, max_nb_layers=3, min_filter_size=32, max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_3_ONON_32_64', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    # # ######################## CNN 1-3 N N O N 32 64 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=1, max_nb_layers=3, min_filter_size=32, max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_3_NNON_32_64', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    # # ######################## CNN 1-3 O N N O 32 64 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=False, use_l1l2_output=True,
    #                                    min_nb_layers=1, max_nb_layers=3, min_filter_size=32, max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_3_ONNO_32_64', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc