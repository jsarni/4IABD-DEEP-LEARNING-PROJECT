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
    #declaration de la structure:
    #cnn_mod=CNNStructurer()
    # res = nb_maxPooling2D_usedmax(32, 4)
    # print('nb max autorisés', res)
    # res = nb_maxPooling2D_usedmax(32, 3)
    # print('nb max autorisés', res)
    # res = nb_maxPooling2D_usedmax(32, 2)
    # print('nb max autorisés', res)
    # res = nb_maxPooling2D_usedmax(32, 32)
    # print('nb max autorisés', res)
    # res = nb_maxPooling2D_usedmax(32, 64)
    # print('nb max autorisés', res)
    # model=models.Sequential()
    # model.add(layers.Conv2D(128,(3,3),padding='same',activation='selu',input_shape=(32,32,3),name='conv0'))
    # model.add(layers.MaxPool2D(3,3))
    # model.add(layers.Conv2D(64, (3, 3), padding='same', activation='selu', name='conv4'))
    # model.add(layers.Conv2D(32, (3, 3), padding='same', activation='selu', name='conv5'))
    # model.add(layers.MaxPool2D(3, 3))
    # model.add(layers.Conv2D(32, (3, 3), padding='same', activation='selu', name='conv8'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(10, activation='softmax', name='conv6'))
    # model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    # model.summary()
    # model.fit(train_images,train_labels,epochs=25,batch_size=512,validation_data=(test_images,test_labels),verbose=2)

    #     (cnn_model, cnn_description) =generateCNNModels(
    #                           nb_Conv2D_layers_list =[3,3,4],
    #                           Conv2D_layers_size_list = [[(32, 3), (32, 3), (64, 3)],[(32, 2), (32, 2),(64,2)],[(32, 2), (32, 2),(32,2),(64,2)]],
    #                           Conv2D_activation_list= ['relu','relu','relu'],
    #                           output_activation_list= ['softmax','softmax','softmax'],
    #                           MaxPooling2D_use_list = [True,True,True],
    #                           MaxPooling2D_Position_list =[[2, 3],[1,2],[1,3]],
    #                           MaxPooling2D_values_list= [3,2,2],
    #                           use_dropout_list=[False,True,True],
    #                           dropout_indexes_list=[[],[2],[3]],
    #                           dropout_value_list=[[],0.02,0.03],
    #                           use_l1l2_regularisation_Conv2D_layers_list=[True,True,True],
    #                           use_l1l2_regularisation_output_layer_list=[False,False,True],
    #                           l1_value_list=[0.02,0.004,0.002],
    #                           l2_value_list=[0.01,0.002,0.001],
    #                           regulization_indexes_list=[[2],[2],[3]],
    #                           loss_list=['sparse_categorical_crossentropy','sparse_categorical_crossentropy','sparse_categorical_crossentropy'],
    #                           optimizer_list=['Adam','Adam','Adam'],
    #                           metrics_list=[['accuracy'],['accuracy'],['accuracy']]
    # )
    #     (cnn_model1, cnn_description1) = generateCNNModels(
    #         nb_Conv2D_layers_list=[7],
    #         Conv2D_layers_size_list=[[(128,2), (128,2), (128,2), (128,2), (128,2), (64, 2), (32, 2)]],
    #         Conv2D_activation_list=['relu'],
    #         output_activation_list=['softmax'],
    #         MaxPooling2D_use_list=[True],
    #         MaxPooling2D_Position_list=[[2,3,6]],
    #         MaxPooling2D_values_list=[2],
    #         use_dropout_list=[True],
    #         dropout_indexes_list=[[3,5]],
    #         dropout_value_list=[0.02],
    #         use_l1l2_regularisation_Conv2D_layers_list=[True],
    #         use_l1l2_regularisation_output_layer_list=[True],
    #         l1_value_list=[0.002],
    #         l2_value_list=[0.001],
    #         regulization_indexes_list=[[3,4]],
    #         loss_list=['sparse_categorical_crossentropy'],
    #         optimizer_list=['Adam'],
    #         metrics_list=[['accuracy']]
    #     )
    #     (cnn_model2, cnn_description2) = generateCNNModels(
    #         nb_Conv2D_layers_list=[4,5,6],
    #         Conv2D_layers_size_list=[[(256,3), (128,3), (64,3), (32,3)],[(256,3), (256,3), (128,3), (64,3), (64,3)],[(256,2), (256,2), (256,2), (128,2), (128,2), (64,2)]],
    #         Conv2D_activation_list=['relu','relu','relu'],
    #         Conv2D_padding_list=['same','same','same'],
    #         output_activation_list=['softmax','softmax','softmax'],
    #         MaxPooling2D_use_list=[True,True,True],
    #         MaxPooling2D_Position_list=[[1],[4],[2,3,5]],
    #         MaxPooling2D_values_list=[3,3,2],
    #         use_dropout_list=[True,False,True],
    #         dropout_indexes_list=[[2],[],[3]],
    #         dropout_value_list=[0.02,0,0.04],
    #         use_l1l2_regularisation_Conv2D_layers_list=[True,True,False],
    #         use_l1l2_regularisation_output_layer_list=[True,True,False],
    #         l1_value_list=[0.003,0.005,0],
    #         l2_value_list=[0.005,0.004,0],
    #         regulization_indexes_list=[[1],[2,3],[]],
    #         loss_list=['sparse_categorical_crossentropy','sparse_categorical_crossentropy','sparse_categorical_crossentropy'],
    #         optimizer_list=['Adam','Adam','Adam'],
    #         metrics_list=[['accuracy'],['accuracy'],['accuracy']]
    #     )
    #     nb_tested_models = test_models('cnn', cnn_model2, cnn_description2, train_images, train_labels, test_images, test_labels,epochs_p=epochs, batch_size_p=512)
    #     print("END : Number of tested models = " + str(nb_tested_models))
    #
    #     model=models.Sequential()
    #     model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(32,32,3)))
    #     model.add(layers.MaxPool2D(3,3))
    #     model.add(layers.Conv2D(32,(3,3),activation='relu',padding='same',kernel_regularizer=l1_l2(0.001,0.002)))
    #     model.add(layers.MaxPool2D(3, 3))
    #     model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l1_l2(0.001, 0.002)))
    #     model.add(layers.MaxPool2D(3, 3))
    #     model.add(layers.Dense(10,activation='softmax'))
    #     model.compile(optimizers='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    #     model.summary()
    #     plot_model(model,'simple_model{}.png')
    #     history=model.fit(train_images,train_labels,epochs=25,batch_size=256,validation_data=(test_images,test_labels))7

    #----------------------------------------------------------------------------------------------------------------------------------------------------------#
    """
    contexte: on va démarrer de modèles simple ( moins de couches, pas dégularisation pas de dropout):
    N = non ,
    O= oui,
    N O O = non dropout, oui regularization conv, oui regul output
    """
    ##32-64##

    # ############################## 3 to 6 layers  N N N 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_N_N_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N N 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_N_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc
    # ############################## 3 to 6 layers  O O N 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=True,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_O_N_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N O 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  N N O 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_N_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N N 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  N O O 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_O_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O O O 32 64 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_O_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # #-----------------------------------------------------------------------------------------------------------------------------------------------------------------#
    clear_session()
    #32-128##
    ############################## 3 to 6 layers  O N N 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_OONN_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc
    # ############################## 3 to 6 layers  O O N 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=True,use_l1l2_conv=True,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_OOON_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc
    #
    # ############################## 3 to 6 layers  O N O 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=True,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_OONO_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc
    #
    # ############################## 3 to 6 layers  N N O 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_ONNO_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc
    #
    # ############################## 3 to 6 layers  O N O 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=True,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_OONO_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc
    #
    # ############################# 3 to 6 layers  N O O 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_ONOO_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256)
    #     del model
    #     del desc

    ############################## 3 to 6 layers  O O O 32 128 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_O_O_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    #     # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    clear_session()
    ##32-256##
    # ############################## 3 to 6 layers  N N N 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_N_N_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N N 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_N_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc
    # ############################## 3 to 6 layers  O O N 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=True,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_O_N_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N O 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_O_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  N N O 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=False,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_N_O_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################## 3 to 6 layers  O N N 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_N_O_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    ############################## 3 to 6 layers  N O O 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_N_O_O_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512)
    #     del model
    #     del desc

    # ############################# 3 to 6 layers  O O O 32 256 #########################
    # for i in range(3):
    #     struct =generateRandoCNNStruc(use_dropout=True,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_O_O_O_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=2048)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # clear_session()
    #
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=False,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=64)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_O_O_O_32_64', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=1024,save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_O_N_O_N_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,batch_size_p=1024,save_image=True)
    #     del model
    #     del desc
    # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=False,use_dropout=True,use_l1l2_conv=False,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_NONN_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs, batch_size_p=256,save_image=True)
    #     del model
    #     del desc
    # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_ONNO_32_128', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256,save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_ONNO_32_256', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=256,save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=True,min_nb_layers=3,max_nb_layers=3,min_filter_size=32,max_filter_size=512)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_3_ONNO_32_512', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512,save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # ##Scirpt du 26 01 2020 :
    # # ######################## CNN 3-6 O N O N 32 512 #########################
    # for i in range(3):
    #     clear_session()
    #     struct =generateRandoCNNStruc(use_maxpool=True,use_dropout=False,use_l1l2_conv=True,use_l1l2_output=False,min_nb_layers=3,max_nb_layers=6,min_filter_size=32,max_filter_size=512)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_6_ONON_32_512', model, desc, train_images, train_labels, test_images, test_labels, epochs_p=epochs,
    #                 batch_size_p=512,save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # epochs=[100]
    # ##Scirpt du 26 01 2020 :
    # # ######################## CNN 3-8 O N O N 32 512 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=512)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_8_ONON_32_512', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=512, save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#

    ##Scirpt du 26 01 2020 :

    ##Scirpt du 26 01 2020 :
    # ######################## CNN 3-8 O O O O 32 256 #########################
    # à lancer le 26 01 2020:
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_8_OOOO_32_256', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # ##Scirpt du 26 01 2020 :
    # # ######################## CNN 3-8 N N O N 32 256 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=3, max_nb_layers=8, min_filter_size=32, max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_8_NNON_32_256', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=512, save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # ##Scirpt du 26 01 2020 :
    # # ######################## CNN 3-12 N O O N 32 256 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=False, use_dropout=True, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=3, max_nb_layers=6, min_filter_size=32, max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_3_12_NOON_32_256', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=512, save_image=True)
    #     del model
    #     del desc
    # # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # (modelcnn,desccnn)=generateCNNModels([7],[[(128,3),(128,3),(128,3),(64,3),(64,3),(32,3),(32,3)]],['relu'],['softmax'],['same'],[True],[[1,2,5]],[3],[True],[[0]],[0.25],[True],[True],[0.002],[0.003],[[0,4,6]],['sparse_categorical_crossentropy'],['Adam'],[['accuracy']])
    #
    # test_models('cnn_7_7_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=128, save_image=True)
    # clear_session()
    # (modelcnn, desccnn) = generateCNNModels([7], [[(128, 3), (128, 3), (64, 3), (64, 3), (32, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 2, 5]], [3], [True], [[0]],
    #                                         [0.25], [True], [False], [0.002], [0.003], [[0, 4, 6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_7_7_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=128, save_image=True)
    # clear_session()
    # (modelcnn, desccnn) = generateCNNModels([7], [[(256, 3), (64, 3), (128, 3), (64, 3), (64, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 2, 5]], [3], [True], [[0]],
    #                                         [0.25], [True], [True], [0.002], [0.003], [[0, 4, 6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_7_7_OOOO_32_256', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=128, save_image=True)
    # clear_session()
    # (modelcnn, desccnn) = generateCNNModels([7], [[(128, 3), (128, 3), (128, 3), (64, 3), (64, 3), (32, 3), (32, 3)]],
    #                                         ['selu'], ['softmax'], ['same'], [True], [[1, 2, 5]], [3], [True], [[0]],
    #                                         [0.25], [True], [True], [0.002], [0.003], [[0, 4, 6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_7_7_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=128, save_image=True)
    # clear_session()
    # (modelcnn, desccnn) = generateCNNModels([7], [[(128, 3), (64, 3), (64, 3), (64, 3), (32, 3), (32, 3), (32, 3)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 2, 5]], [3], [True], [[0]],
    #                                         [0.15], [True], [True], [0.001], [0.003], [[0, 4, 6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_7_7_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=128, save_image=True)
    # 27/01/2020----19h50
    # clear_session()
    # epochs=[150]
    # (modelcnn, desccnn) = generateCNNModels([8],[[(128, 4),(128, 4),(128, 4),(64, 4), (64, 4) ,(64, 4) ,(32, 4) ,(32, 4)]],
    #                                         ['relu'],['softmax'],['same'],[True],[[1,4]],[4],[True],[[0,4,6]],
    #                                         [0.025],[True],[True],[0.012],[0.025],[[6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=256, save_image=True)
    # # ######################## CNN 3-12 N O O N 32 256 #########################
    # for i in range(3):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=True, use_l1l2_conv=True, use_l1l2_output=False,
    #                                    min_nb_layers=8, max_nb_layers=8, min_filter_size=32, max_filter_size=128)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_8_8_NOON_32_128', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=256, save_image=True)
    #     del model
    #     del desc
    #
    # clear_session()
    # (modelcnn, desccnn) = generateCNNModels([8], [[(64, 4),(128, 4),(128, 4),(64, 4),(64, 4),(64, 4),(32, 4),(32, 4)]],
    #                                         ['relu'], ['softmax'], ['same'], [True], [[1, 4]], [4], [True], [[0, 4, 6]],
    #                                         [0.025], [True], [True], [0.012], [0.025], [[6]],
    #                                         ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])
    #
    # test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
    #             epochs_p=epochs,
    #             batch_size_p=256, save_image=True)
    # clear_session()
    #testes 28/01/2020 08h22
    epochs=[200]
    (modelcnn, desccnn) = generateCNNModels([8], [[(128, 4),(128, 4),(64, 4),(64, 4),(64, 4),(32, 4),(32, 4),(32, 4)]],
                                            ['relu'], ['softmax'], ['same'], [True], [[1, 4]], [4], [True], [[0, 4, 6]],
                                            [0.025], [True], [True], [0.012], [0.025], [[2,7]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])

    test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=epochs,
                batch_size_p=256, save_image=True)
    clear_session()

    (modelcnn, desccnn) = generateCNNModels([8], [
        [(128, 4), (128, 4), (64, 4), (64, 4), (64, 4), (32, 4), (32, 4), (32, 4)]],
                                            ['relu'], ['softmax'], ['same'], [True], [[1, 4]], [4], [True], [[0, 4, 6]],
                                            [0.028], [True], [True], [0.012], [0.025], [[2,3,7]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])

    test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=epochs,
                batch_size_p=256, save_image=True)
    clear_session()

    (modelcnn, desccnn) = generateCNNModels([8], [
        [(128, 4), (128, 4), (64, 4), (64, 4), (64, 4), (32, 4), (32, 4), (32, 4)]],
                                            ['relu'], ['softmax'], ['same'], [True], [[1, 5]], [4], [True], [[0, 4, 6]],
                                            [0.025], [True], [True], [0.012], [0.025], [[3,7]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])

    test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=epochs,
                batch_size_p=256, save_image=True)
    clear_session()

    (modelcnn, desccnn) = generateCNNModels([8], [
        [(128, 4), (128, 4), (64, 4), (64, 4), (64, 4), (32, 4), (32, 4), (32, 4)]],
                                            ['relu'], ['softmax'], ['same'], [True], [[1, 5]], [4], [True], [[0, 4, 6]],
                                            [0.025], [True], [True], [0.012], [0.035], [[3, 7]],
                                            ['sparse_categorical_crossentropy'], ['Adam'], [['accuracy']])

    test_models('cnn_8_8_OOOO_32_128', modelcnn, desccnn, train_images, train_labels, test_images, test_labels,
                epochs_p=epochs,
                batch_size_p=256, save_image=True)
    clear_session()

    for i in range(3):
        clear_session()
        struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=True, use_l1l2_conv=True, use_l1l2_output=False,
                                       min_nb_layers=8, max_nb_layers=8, min_filter_size=32, max_filter_size=128)
        model = [create_CNN_model(struct)]
        desc = [getcnnStructAsString(struct)]
        test_models('cnn_8_8_NOON_32_128', model, desc, train_images, train_labels, test_images, test_labels,
                    epochs_p=epochs,
                    batch_size_p=256, save_image=True)
        del model
        del desc

    # for i in range(2):
    #     clear_session()
    #     struct = generateRandoCNNStruc(use_maxpool=True, use_dropout=False, use_l1l2_conv=True, use_l1l2_output=True,
    #                                    min_nb_layers=1, max_nb_layers=5, min_filter_size=64, max_filter_size=256)
    #     model = [create_CNN_model(struct)]
    #     desc = [getcnnStructAsString(struct)]
    #     test_models('cnn_1_5_ONOO_64_256', model, desc, train_images, train_labels, test_images, test_labels,
    #                 epochs_p=epochs,
    #                 batch_size_p=128, save_image=True)
    #     del model
    #     del desc
    # # #------------------------------------------------------------------------------------------------------------------------------------------------------------#
