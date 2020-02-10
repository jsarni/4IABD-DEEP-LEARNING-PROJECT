from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
import tensorflow as tf

from cifar10.models.CNN import *
from cifar10.models.ModelTester import test_models

if __name__ == "__main__":
    #déclaration des param:
    epochs=[40]
    #importation des données:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    #définition des noms des classes:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


    # (modelcnn, desccnn) = generateCNNModels([8], [
    #     [(128, 4), (128, 4), (64, 4), (64, 4), (64, 4), (32, 4), (32, 4), (32, 4)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 4]], [4], [True], [[0, 4, 6]],
    #                                         [0.044], [True], [True], [0.013], [0.040], [[2,7]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #
    #
    # #31/01/2020
    # epochs=[40]
    # (modelcnn, desccnn) = generateCNNModels([8], [
    #     [(32, 3), (64, 3) ,(32, 3), (128, 3) ,(64, 3) ,(32, 3) ,(64, 3) ,(32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[8]], [3], [False], [[0]],
    #                                         [0], [True], [False], [0.047], [0.001], [[6,7]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    # test_models('cnn_3_12_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #
    #

    # epochs = [50]
    #
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=True, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=6, max_nb_layers=10, min_filter_size=32, max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_8_8_NOON_32_128', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=128, save_image=True)
    #     del model
    #     del desc
    #

    # new huge model on 04/02/2020:
    # for e in (40,45,50,55,60):
    #     (modelcnn, desccnn) = generateCNNModels([15], [
    #     [(64, 2), (64, 2), (64, 2), (64, 2), (64, 2), (128, 2), (128, 2), (128, 2), (64, 2), (64, 2), (32, 2), (32, 2),
    #      (32, 2), (32, 2), (32, 2)]],
    #                                         ['selu'], ['softmax'], ['same'], [True], [[2,4, 8, 12]], [2], [True],
    #                                         [[0, 6, 14]],
    #                                         [0.70], [True], [False], [0.045], [0.099], [[10, 12]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_1515_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=[e],
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #;relu;softmax;same;True;3;4 5;False;;0.0;True;True;0.002;0.033;4 1 4;

    #8/2:
    # for e in (25,30,35,40,45,50):
    #     (modelcnn, desccnn) = generateCNNModels([5], [
    #         [(64, 3), (128, 3),(64, 3), (32, 3) ,(32, 3)]],
    #                                             ['selu'], ['softmax'], ['same'], [True], [[2,4]], [3], [True], [[0]],
    #                                             [0.43], [True], [False], [0.045], [0.099], [[1,3]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_5_5_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=[e],
    #                 batch_size_p=256, save_image=True)
    #     clear_session()

    # for e in (25,30,35,40,45):
    #     (modelcnn, desccnn) = generateCNNModels([4], [
    #         [(64, 3), (128, 3),(64, 3),(32, 3)]],
    #                                             ['relu'], ['softmax'], ['same'], [True], [[2,4]], [3], [True], [[0]],
    #                                             [0.25], [True], [False], [0.030], [0.090], [[1,3]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_5_5_ONON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=[e],
    #                 batch_size_p=256, save_image=True)
    #     clear_session()

    #tested on 8/2/2020 à 12h08
    #5;(64, 3) (128, 3) (64, 3) (32, 3) (32, 3);selu;softmax;same;True;3;2 4;False;0;0;True;False;0.005;0.06;1 3
    e=[45]
    for i in (0.15,0.25,0.35,0.45):
        (modelcnn, desccnn) = generateCNNModels([5], [
                    [(64, 3), (128, 3),(64, 3), (32, 3) ,(32, 3)]],
                                                        ['selu'], ['softmax'], ['same'], [True], [[2,4]], [3], [False], [[0]],
                                                        [0], [True], [True], [i], [0.099], [[1,3]],
                                                        ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_5_5_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                            epochs_p=[e],
                            batch_size_p=256, save_image=True)
        clear_session()

    for e in (30,35,40,45,50):
        (modelcnn, desccnn) = generateCNNModels([6], [
            [(64, 3), (128, 3), (128, 3) ,(64, 3), (32, 3) ,(32, 3)]],
                                                ['relu'], ['softmax'], ['same'], [True], [[4,5]], [3], [True], [[0]],
                                                [0.35], [True], [True], [0.05], [0.095], [[1,4]],
                                                ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_6_6_OOOO_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                    epochs_p=[e],
                    batch_size_p=256, save_image=True)
        clear_session()

    for e in (25,30,35,40,45,50):
        (modelcnn, desccnn) = generateCNNModels([7], [
            [(64, 3), (128, 3), (128, 3) ,(64, 3),(64, 3), (32, 3) ,(32, 3)]],
                                                ['selu'], ['softmax'], ['same'], [True], [[4,5]], [3], [True], [[0,3]],
                                                [0.25], [True], [False], [0.035], [0.098], [[1,4,6]],
                                                ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_7_7_ONON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                    epochs_p=[e],
                    batch_size_p=256, save_image=True)
        clear_session()
    #test 9:
    for e in (25,30,35,40,45,50):
        (modelcnn, desccnn) = generateCNNModels([9], [
            [(64, 3),(64, 3), (128, 3), (128, 3) ,(64, 3),(64, 3), (32, 3), (32, 3) ,(32, 3)]],
                                                ['relu'], ['softmax'], ['same'], [True], [[3,5,8]], [3], [True], [[0,7]],
                                                [0.33], [True], [False], [0.035], [0.099], [[1,4,6]],
                                                ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_9_9_OOOO_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                    epochs_p=[e],
                    batch_size_p=256, save_image=True)
        clear_session()




    # # 28/01/2020 à 18h35:
    #
    # (modelcnn, desccnn) = generateCNNModels([12], [
    #     [(128, 4), (128, 4), (128, 4), (64, 4), (64, 4), (64, 4), (64, 4), (64, 4), (32, 4), (32, 4), (32, 4),
    #      (32, 4)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 6]], [4], [True], [[0, 6]],
    #                                         [0.046], [True], [False], [0.018], [0.040], [[4, 9, 11]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    # test_models('cnn_3_12_OOON_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=256, save_image=True)
    #
    # clear_session()


