from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers
from tensorflow.keras.backend import clear_session
import tensorflow as tf

from cifar10.models.CNN import *
from cifar10.models.ModelTester import test_models

if __name__ == "__main__":
    #déclaration des param:
    e=[40]
    #importation des données:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    #définition des noms des classes:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    #test du 5:
    # for i in (35,45,55):
    #     (modelcnn, desccnn) = generateCNNModels([5], [
    #         [(256, 4), (128, 4), (128, 4), (64, 4), (32, 4)]],
    #                                             ['relu'], ['softmax'], ['same'], [True], [[2,4]], [4], [False], [[0]],
    #                                             [0], [True], [False], [0.012], [0.050], [[0,1,3]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_5_5_ONOO_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=[i], batch_size_p=256, save_image=True)
    #     clear_session()
    #5;;selu;softmax;same;True;3;2 4;False;0;0;True;False;0.005;0.06;1 3;sparse_categorical_crossentropy;str;accuracy;35
    #à lancer le 9/02 21h
    for i in (128,256,512):
        (modelcnn, desccnn) = generateCNNModels([5], [
        [(64, 3) ,(128, 3) ,(64, 3),(32, 3),(32, 3)]],
                                            ['selu'], ['softmax'], ['same'], [True], [[2, 4]], [3], [False], [[0]],
                                            [0], [True], [False], [0.005], [0.050], [[ 1,3]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
        test_models('cnn_my_best', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=e, batch_size_p=i, save_image=True,save_model=True)
        clear_session()
    # # test du 6:
    # (modelcnn, desccnn) = generateCNNModels([6], [
    #     [(64, 3), (128, 3), (64, 3),(32, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [False], [[]], [0], [False], [[0]],
    #                                         [0], [False], [False], [0], [0], [[]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    # test_models('cnn_6_6_NNNN_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=e,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #
    # #test du 8
    # (modelcnn, desccnn) = generateCNNModels([8], [
    #     [(64, 3), (128, 3), (128, 3), (64, 3), (64, 3), (32, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [False], [[]], [0], [False], [[0]],
    #                                         [0], [False], [False], [0], [0], [[]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    # test_models('cnn_8_8_NNNN_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=e,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #test du 12:
    #10;;relu;softmax;same;True;4;1 4;True;0 5;0.01;True;True;0.002;0.005;3 8;sparse_categorical_crossentropy;str;accuracy;30;8435598;0.8558;0.7152;20200127

    # #test 10 ONOO
    # for i in (0.15,0.25,0.35):
    #     (modelcnn, desccnn) = generateCNNModels([10], [
    #         [(128, 4) ,(128, 4) ,(128, 4) ,(64, 4) ,(64, 4), (64, 4) ,(64, 4), (32, 4) ,(32, 4) ,(32, 4)]],
    #                                             ['relu'], ['softmax'], ['same'], [True], [[2,7]], [4], [False], [[0]],
    #                                             [0], [True], [True], [0.15], [i], [[1,5]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_10_ONOO_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=e,
    #                 batch_size_p=256, save_image=True)
    #     clear_session()

    #8;(64, 3) (32, 3) (64, 3) (32, 3) (32, 3) (64, 3) (32, 3) (64, 3);selu;softmax;same;False;3;;False;;0.0;True;False;0.086;0.03;8 2 8 6 5 8 5;sparse_categorical_crossentropy;str;['accuracy'];30;3046826;0.36302;0.3629;20200126
    # test du 7:
    # for i in (32,64,128):
    #     (modelcnn, desccnn) = generateCNNModels([7], [
    #     [(64, 3),(64, 3), (128, 3), (i, 3),(i, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [False], [[]], [0], [False], [[0]],
    #                                         [0], [False], [False], [0], [0], [[]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_7_7_NNNN_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=e,
    #             batch_size_p=256, save_image=True)
    #     clear_session()
    #
    #
    # #8 NNON
    # for i in (0.05,0.015,0.025):
    #     (modelcnn, desccnn) = generateCNNModels([8], [
    #         [(128, 4) ,(64, 4) ,(64, 4) ,(64, 4), (64, 4), (32, 4) ,(32, 4) ,(32, 4)]],
    #                                             ['relu'], ['softmax'], ['same'], [True], [[2,4]], [4], [False], [[0]],
    #                                             [0], [True], [False], [0.05], [i], [[0,5,7]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_8_8_NNON_32_64', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=e, batch_size_p=256, save_image=True)
    #     clear_session()
    # # #tested12 NNNN:
    # # (modelcnn, desccnn) = generateCNNModels([12], [
    # #     [(64, 3), (64, 3), (64, 3), (128, 3), (128, 3), (64, 3), (64, 3), (32, 3), (32, 3), (32, 3), (32, 3),
    # #      (32, 3)]],
    # #                                         ['relu'], ['softmax'], ['same'], [False], [[]], [0], [False], [[0]],
    # #                                         [0], [False], [False], [0], [0], [[]],
    # #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    # # test_models('cnn_1212_NNNN_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    # #             epochs_p=e,
    # #             batch_size_p=256, save_image=True)
    # # clear_session()
    # #test 15 ONON
    # for i in (0.03,0.04,0.05):
    #     (modelcnn, desccnn) = generateCNNModels([15], [
    #     [(64, 2) ,(64, 2), (64, 2) ,(64, 2) ,(64, 2), (128, 2), (128, 2) ,(128, 2), (64, 2), (64, 2), (32, 2) ,(32, 2), (32, 2), (32, 2) ,(32, 2)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[2,4,8,12]], [2], [False], [[0]],
    #                                         [0], [True], [False], [i], [0.95], [[6,14]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_15_ONOO_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=e,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    # #tested 15 NNNN
    # # Test NNNN:
    # for i in (30,35,40):
    #     (modelcnn, desccnn) = generateCNNModels([15], [
    #     [(64, 2) ,(64, 2), (64, 2) ,(64, 2) ,(64, 2), (128, 2), (128, 2), (128, 2) ,(64, 2), (64, 2) ,(32, 2) ,(32, 2) ,(32, 2), (32, 2) ,(32, 2)]],
    #                                         ['relu'], ['softmax'], ['same'], [False], [[]], [0], [False], [[0]],
    #                                             [0], [False], [False], [0], [0], [[]],
    #                                             ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #     test_models('cnn_15_NNNN_32128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=[i],
    #                 batch_size_p=256, save_image=True)
    #     clear_session()