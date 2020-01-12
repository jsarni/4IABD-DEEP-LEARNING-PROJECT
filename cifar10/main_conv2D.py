from __future__ import absolute_import, division, print_function, unicode_literals
from cifar10.DatasetLoader import *
from tensorflow.keras import layers, activations,models,optimizers,metrics,datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pk
from datetime import datetime
from tensorflow.keras.regularizers import *
from tensorflow.keras.utils import *



class CNNStructurer:

    def __init__(self):
        self.name = "cnn" #nom de la structure
        self.nb_Conv2D_layers = 3 #nombre de couches cachées
        self.Conv2D_size_layers = [(32,3),(32,3),(64,3)]  # [input,filter_dimension] dans l'appel on utilisera un couple ( filter_dimension,filter_dimension)
        self.Conv2D_activation ='relu'
        self.MaxPooling2D_use = True
        self.MaxPooling2D_Position = [2,3] #Positionnement des couches Max2Pooling
        self.MaxPooling2D_values = 2   #valeur du filtre Max2Pooling
        self.nb_hidden_layers = 3 #nombre de Denses cachées
        self.layers_size = []
        self.layers_activation = 'relu' #activation
        self.output_activation = 'softmax' #activation output
        self.use_dropout = True
        self.dropout_indexes = [2,3]
        self.dropout_value = 0.01
        self.use_l1l2_regularisation_Convolution_layers = True
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = True
        self.l1_value = 0.02
        self.l2_value = 0.03
        self.regul_kernel_indexes = [1,2]
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = 'Adam'
        self.metrics = ['accuracy']


def create_CNN_model(cnn_struct:CNNStructurer):
    #format des données
    inputshape=(32,32,3)
    #vérification principale:
    assert cnn_struct.nb_hidden_layers == len(cnn_struct.layers_size) and (cnn_struct.nb_Conv2D_layers==len(cnn_struct.Conv2D_size_layers)), "CNNStructurerError: CNN number of layers (nb_hidden_layers) is different of the total layers sizes number (layer_size) "
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
                                name="Input_with_l1_l2"
                                )
                  )
    else:
        model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[0][0],
                                (cnn_struct.Conv2D_size_layers[0][1], cnn_struct.Conv2D_size_layers[0][1]),
                                activation=cnn_struct.Conv2D_activation, input_shape=inputshape,
                                name="Input"
                                )
                  )

    #Création du reste des couches de convolution Conv2D et les MaxPooling2D
    for i in range(len(cnn_struct.Conv2D_size_layers)):
        #test if the layer has a MaxPooling
        if(cnn_struct.MaxPooling2D_use and i in cnn_struct.MaxPooling2D_Position):

            model.add(layers.MaxPool2D((cnn_struct.MaxPooling2D_values,cnn_struct.MaxPooling2D_values)))
        #voir si la couche admet une régularisation:
        if  (i in cnn_struct.regul_kernel_indexes):
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1],cnn_struct.Conv2D_size_layers[i][1]),
                                    kernel_regularizer=l1_l2(cnn_struct.l1_value, cnn_struct.l2_value),
                                    activation=cnn_struct.Conv2D_activation,
                                    name=f"Conv_with_l1_l2_{i}"
                                    )
                      )
        else:
            model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],
                                    (cnn_struct.Conv2D_size_layers[i][1], cnn_struct.Conv2D_size_layers[i][1]),
                                    kernel_regularizer=l1_l2(cnn_struct.l1_value, cnn_struct.l2_value),
                                    activation=cnn_struct.Conv2D_activation,
                                    name=f"Conv_{i}"
                                    )
                      )
        # add dropout layer:
        if (i in cnn_struct.dropout_indexes):
                model.add(layers.Dropout(cnn_struct.dropout_value))
    """#Création de la dernière couche de convolution:
    i=cnn_struct.nb_Conv2D_layers

    model.add(layers.Conv2D(cnn_struct.Conv2D_size_layers[i][0],(cnn_struct.layers_size[i][1], cnn_struct.layers_size[i][1]),activations=cnn_struct.Conv2D_activation)
        # Hidden layers L1L2 regularisation
        if cnn_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in cnn_struct.regulization_indexes):
            model.add(layers.Dense(cnn_struct.layers_size[i],
                            activation=cnn_struct.layers_activation,
                            kernel_regularizer=L1L2(l1=cnn_struct.l1_value, l2=cnn_struct.l2_value),
                            name=f"dense_l1l2_{i}"
                            )
                      )
        else:
            model.add(Dense(cnn_struct.layers_size[i],
                            activation=cnn_struct.layers_activation,
                            name=f"dense_{i}"
                            )
                      )
        if cnn_struct.use_dropout and ((i + 1) in cnn_struct.dropout_indexes):
            model.add(Dropout(cnn_struct.dropout_value, name=f"dropout_{i}"))
    """
    ####
    # ajout d'un flatten:
    model.add(layers.Flatten())


    # Hidden layers L1L2 regularisation
    for i in range(len(cnn_struct.nb_hidden_layers)):
        if cnn_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in cnn_struct.regulization_indexes):
            model.add(layers.Dense(cnn_struct.layers_size[i],
                            activation=cnn_struct.layers_activation,
                            kernel_regularizer=L1L2(l1=cnn_struct.l1_value, l2=cnn_struct.l2_value),
                            name=f"dense_l1l2_{i}"
                            )
                      )
        else:
            model.add(layers.Dense(cnn_struct.layers_size[i],
                            activation=cnn_struct.layers_activation,
                            name=f"dense_{i}"
                            )
                      )
        if cnn_struct.use_dropout and ((i + 1) in cnn_struct.dropout_indexes):
            model.add(layers.Dropout(cnn_struct.dropout_value, name=f"dropout_{i}"))

        # Output L1L2 regularisation


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
    # Output L1L2 regularisation
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

    model.compile(loss=cnn_struct.loss, optimizer=cnn_struct.optimizer, metrics=cnn_struct.metrics)

    return model


if __name__ == "__main__":
    #importation des données:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    #définition des noms des classes:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    #declaration de la structure:
    cnn_mod=CNNStructurer()
    #creation du modèle:
    model=create_CNN_model(cnn_mod)
    # #création du modèle:
    # model = models.Sequential()
    # #ajouter des couches de convolution:
    # model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(32, 32, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    #
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    #
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #
    #
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # #applatir les données:
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))
    # #afficher les détails :
    # model.summary()
    # #entrainer le model:
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    #fit du modele:
    history = model.fit(train_images, train_labels, epochs=8,batch_size=5024,
                        validation_data=(test_images, test_labels))
    print(history)

    #Evaluer le model:
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('résultat accuracy',test_acc)
    print('résultat loss', test_loss)
    #Sauvegarder les résultats:

    #Affichage des résultats:
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    #affichage du model:
    plot_model(model, "./test.png")
    """#logger les résultats:
    dict_save={'Accuracy':history.history['accuracy'],'val-accuracy':history.history['val_accuracy'],'date-time':datetime.now()}
    logging_file =open("./logfile","wb")
    log=pk.dump(dict_save,logging_file)
    logging_file.close()
    """


