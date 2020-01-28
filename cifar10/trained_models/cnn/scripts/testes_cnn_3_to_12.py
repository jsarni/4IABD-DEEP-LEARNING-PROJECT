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
    #28/01/2020 à 18h35:

    (modelcnn, desccnn) = generateCNNModels([12], [
        [(128, 4),(128, 4), (128, 4), (64, 4), (64, 4),(64, 4),(64, 4), (64, 4), (32, 4), (32, 4), (32, 4), (32, 4)]],
                                            ['relu'], ['softmax'], ['same'], [True], [[1, 6]], [4], [True], [[0,6]],
                                            [0.038], [True], [False], [0.012], [0.037], [[4,9, 11]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    test_models('cnn_3_12_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=epochs,
                batch_size_p=256, save_image=True)
    clear_session()

    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=True,
                                       min_nb_layers=6, max_nb_layers=12, min_filter_size=32, max_filter_size=128)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_8_8_NOON_32_128', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc



