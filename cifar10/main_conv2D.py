from __future__ import absolute_import, division, print_function, unicode_literals
from cifar10.DatasetLoader import *
from cifar10.models.structurer.CNNStructurer import *
from tensorflow.keras import layers, activations,models,optimizers,metrics,datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pk
from datetime import datetime
from tensorflow.keras.regularizers import *
from tensorflow.keras.utils import *
from random import  randint,choice

def create_CNN_model(cnn_struct:CNNStructurer):
    #format des données
    inputshape=(32,32,3)
    #vérification principale:
    assert (cnn_struct.nb_Conv2D_layers==len(cnn_struct.Conv2D_size_layers)), "CNNStructurerError: CNN number of layers  is different of the total layers sizes number "
    #Déclaration du modèle
    model = models.Sequential()
    #création de la première couche de convolution avec vérification si la première couche admet une régularisation:
    if cnn_struct.use_l1l2_regularisation_Convolution_layers and (0 in cnn_struct.regul_kernel_indexes):
        model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[0][0]
                                ,(cnn_struct.Conv2D_size_layers[0][1],
                                  cnn_struct.Conv2D_size_layers[0][1]),
                                kernel_regularizer=l1_l2(cnn_struct.l1_value,cnn_struct.l2_value),
                                activation=cnn_struct.Conv2D_activation,
                                input_shape=inputshape,
                                padding=cnn_struct.Conv2D_padding,
                                name="Conv_0_with_l1_l2"
                                )
                  )
    else:
        model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[0][0],
                                (cnn_struct.Conv2D_size_layers[0][1], cnn_struct.Conv2D_size_layers[0][1]),
                                activation=cnn_struct.Conv2D_activation,
                                input_shape=inputshape,
                                padding=cnn_struct.Conv2D_padding,
                                name="Conv_0"
                                )
                  )
    #tester si l'input admet un DropOut:
    if(cnn_struct.use_dropout and 0 in cnn_struct.dropout_indexes):
        model.add(layers.Dropout(cnn_struct.dropout_value))
    #Création du reste des couches de convolution Conv2D et les MaxPooling2D
    for i in range(1,len(cnn_struct.Conv2D_size_layers)):
        #test if the layer has a MaxPooling
        if(cnn_struct.MaxPooling2D_use and i in cnn_struct.MaxPooling2D_Position):

            model.add(layers.MaxPool2D((cnn_struct.MaxPooling2D_values,cnn_struct.MaxPooling2D_values)))
        #voir si la couche admet une régularisation:
        if  (i+1 in cnn_struct.regul_kernel_indexes):
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1],cnn_struct.Conv2D_size_layers[i][1]),
                                    kernel_regularizer=l1_l2(cnn_struct.l1_value, cnn_struct.l2_value),
                                    activation=cnn_struct.Conv2D_activation,
                                    padding=cnn_struct.Conv2D_padding,
                                    name=f"Conv_with_l1_l2_{i}"
                                    )
                      )
        else:
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1], cnn_struct.Conv2D_size_layers[i][1]),
                                    kernel_regularizer=l1_l2(cnn_struct.l1_value, cnn_struct.l2_value),
                                    activation=cnn_struct.Conv2D_activation,
                                    padding=cnn_struct.Conv2D_padding,
                                    name=f"Conv_{i}"
                                    )
                      )
        # add dropout layer:
        if (cnn_struct.use_dropout and i in cnn_struct.dropout_indexes):
                model.add(layers.Dropout(cnn_struct.dropout_value))
    #fin de la boucle
    # ajout d'un flatten:
    model.add(layers.Flatten())


    if cnn_struct.use_l1l2_regularisation_output_layer:
        model.add(layers.Dense(10,
                        activation=cnn_struct.output_activation,
                        kernel_regularizer=L1L2(l1=cnn_struct.l1_value, l2=cnn_struct.l2_value),
                        name="output_l1l2"
                        )
                  )
    else:
        model.add(layers.Dense(10,
                        activation=cnn_struct.output_activation,
                        name="output"
                        )
                  )
    ####


    model.compile(loss=cnn_struct.loss, optimizer=cnn_struct.optimizer, metrics=cnn_struct.metrics)
    model.summary()
    plot_model(model,"./trained_models/cnn/model_architecture.png")
    return model



######################################################création d'une strucure cnn####################################"

def generateCNNModels(    nb_Conv2D_layers_list: list,
                          Conv2D_layers_size_list: list,
                          Conv2D_activation_list: list,
                          output_activation_list: list,
                          Conv2D_padding_list: list,
                          MaxPooling2D_use_list : list,
                          MaxPooling2D_Position_list :list,
                          MaxPooling2D_values_list:list,
                          use_dropout_list: list,
                          dropout_indexes_list: list,
                          dropout_value_list: list,
                          use_l1l2_regularisation_Conv2D_layers_list: list,
                          use_l1l2_regularisation_output_layer_list: list,
                          l1_value_list: list,
                          l2_value_list: list,
                          regulization_indexes_list: list,
                          loss_list: list,
                          optimizer_list: list,
                          metrics_list: list):
    assert len(nb_Conv2D_layers_list) == len(Conv2D_layers_size_list)
    assert len(nb_Conv2D_layers_list) == len(Conv2D_activation_list)
    assert len(nb_Conv2D_layers_list) == len(output_activation_list)
    assert len(nb_Conv2D_layers_list) == len(Conv2D_padding_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_use_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_Position_list)
    assert len(nb_Conv2D_layers_list) == len(MaxPooling2D_values_list)
    assert len(nb_Conv2D_layers_list) == len(use_dropout_list)
    assert len(nb_Conv2D_layers_list) == len(dropout_indexes_list)
    assert len(nb_Conv2D_layers_list) == len(dropout_value_list)
    assert len(nb_Conv2D_layers_list) == len(use_l1l2_regularisation_Conv2D_layers_list)
    assert len(nb_Conv2D_layers_list) == len(use_l1l2_regularisation_output_layer_list)
    assert len(nb_Conv2D_layers_list) == len(l1_value_list)
    assert len(nb_Conv2D_layers_list) == len(l2_value_list)
    assert len(nb_Conv2D_layers_list) == len(regulization_indexes_list)
    assert len(nb_Conv2D_layers_list) == len(loss_list)
    assert len(nb_Conv2D_layers_list) == len(optimizer_list)
    assert len(nb_Conv2D_layers_list) == len(metrics_list)

    cnn_models = []
    cnn_descriptions = []

    current_structure = CNNStructurer()

    for i in range(len(nb_Conv2D_layers_list)):
        current_structure.nb_Conv2D_layers   = nb_Conv2D_layers_list[i]
        current_structure.Conv2D_size_layers = Conv2D_layers_size_list[i]
        current_structure.Conv2D_activation  = Conv2D_activation_list[i]
        current_structure.Conv2D_padding=Conv2D_padding_list[i]
        current_structure.output_activation  = output_activation_list[i]
        current_structure.MaxPooling2D_use   = MaxPooling2D_use_list[i]
        current_structure.MaxPooling2D_Position  = MaxPooling2D_Position_list[i]
        current_structure.MaxPooling2D_values  =MaxPooling2D_values_list[i]
        current_structure.use_dropout        = use_dropout_list[i]
        current_structure.dropout_indexes    = dropout_indexes_list[i]
        current_structure.dropout_value      = dropout_value_list[i]
        current_structure.use_l1l2_regularisation_Convolution_layers = use_l1l2_regularisation_Conv2D_layers_list[i]
        current_structure.use_l1l2_regularisation_output_layer = use_l1l2_regularisation_output_layer_list[i]
        current_structure.l1_value = l1_value_list[i]
        current_structure.l2_value = l2_value_list[i]
        current_structure.regulization_indexes = regulization_indexes_list[i]
        current_structure.loss = loss_list[i]
        current_structure.optimizer = optimizer_list[i]
        current_structure.metrics = metrics_list[i]

        cnn_models.append(create_CNN_model(current_structure))
        cnn_descriptions.append(getcnnStructAsString(current_structure))

    return cnn_models, cnn_descriptions
##
def getcnnStructAsString(cnn_structurer):
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(cnn_structurer.nb_Conv2D_layers,
                                                                    " ".join([str(i) for i in cnn_structurer.Conv2D_size_layers]),
                                                                    cnn_structurer.Conv2D_activation,
                                                                    cnn_structurer.output_activation,
                                                                    cnn_structurer.Conv2D_padding,
                                                                    cnn_structurer.MaxPooling2D_use,
                                                                    cnn_structurer.MaxPooling2D_values,
                                                                 " ".join([str(i) for i in cnn_structurer.MaxPooling2D_Position]),
                                                                    cnn_structurer.use_dropout,
                                                                    " ".join([str(i) for i in cnn_structurer.dropout_indexes]),
                                                                    cnn_structurer.dropout_value,
                                                                    cnn_structurer.use_l1l2_regularisation_Convolution_layers,
                                                                    cnn_structurer.use_l1l2_regularisation_output_layer,
                                                                    cnn_structurer.l1_value,
                                                                    cnn_structurer.l2_value,
                                                                    " ".join([str(i) for i in cnn_structurer.regul_kernel_indexes]),
                                                                    cnn_structurer.loss,
                                                                    cnn_structurer.optimizer.__class__.__name__,
                                                                    " ".join([str(i) for i in cnn_structurer.metrics])
                                                                    )

def getRandomModelID():
    uid = random.randint(0, 10000000)
    return "{:07d}".format(uid)
#cette fonction nous renvoi le nombre max de MaxPooling2D qu'on peut appliqeur avec un kernel précis.
def nb_maxPooling2D_usedmax(filter:int,kernel:int):
    res = 0
    if(filter>=kernel):
        res=1
    while (int(filter / kernel) >= kernel):
        res += 1
        filter = int(filter / kernel)
    return res

def generateRandoCNNStruc(use_maxpool=False, use_l1l2_conv=False, use_l1l2_output=False, use_dropout=False, min_nb_layers=3, max_nb_layers=8,min_filter_size=32,max_filter_size=64):
    layers_activations = ['softmax', 'relu', 'softplus', 'selu']
    output_activations = ['softmax']
    kernel_sizes =randint(2,5)
    #génération des filtres des convolutions:
    filters = []
    i=min_filter_size
    while(i<=max_filter_size):
        filters.append(i)
        i=i*2
    #fin de la génération

    metrics = [['accuracy']]
    losses = ['sparse_categorical_crossentropy']
    optimizers = ['Adam']
    nb_layers = randint(min_nb_layers, max_nb_layers)
    use_dropout = use_dropout
    use_l1l2 = use_l1l2_conv
    use_l1l2_output = use_l1l2_output
    dropout_indexes = []
    dropout_value = 0.0
    if use_dropout:
        dropout_indexes_number = randint(0, nb_layers)
        dropout_value = randint(0, 4) / 10
        for j in range(dropout_indexes_number):
            dropout_indexes.append(randint(0, nb_layers))
    l1l2_indexes = []
    l1_value = 0.0
    l2_value = 0.0
    ##faut faire une boucle à la longueur des layers et tirer des layers
    if use_l1l2:
        l1l2_indexes_number = randint(1, nb_layers)
        for j in range(l1l2_indexes_number):
            l1l2_indexes.append(randint(1, nb_layers))
        l1_value = randint(5, 100)/1000
        l2_value = randint(5, 100) / 1000

    maxpool_indexes = []
    if use_maxpool:
        nb_maxpool_layers = randint(0, nb_maxPooling2D_usedmax(32,kernel_sizes))
        for j in range(nb_maxpool_layers):
            maxpool_indexes.append(randint(1,nb_layers))
    filter_size=[]
    for i in range(nb_layers):
        filter_size.append((choice(filters),kernel_sizes))

    struct = CNNStructurer()
    struct.nb_Conv2D_layers = nb_layers
    struct.Conv2D_size_layers =filter_size
    struct.Conv2D_activation = choice(layers_activations)
    struct.output_activation = choice(output_activations)
    struct.use_MaxPooling2D = use_maxpool
    struct.MaxPooling2D_position = maxpool_indexes
    struct.MaxPooling2D_values=kernel_sizes
    struct.use_dropout = use_dropout
    struct.dropout_indexes = dropout_indexes
    struct.dropout_value = dropout_value
    struct.use_l1l2_regularisation_Convolution_layers= use_l1l2_conv
    struct.use_l1l2_regularisation_output_layer = use_l1l2_output
    struct.l1_value = l1_value
    struct.l2_value = l2_value
    struct.regul_kernel_indexes = l1l2_indexes
    struct.loss = choice(losses)
    struct.optimizer = choice(optimizers)
    struct.metrics = [choice(metrics)]
    struct.Conv2D_padding = 'same'

    return struct
