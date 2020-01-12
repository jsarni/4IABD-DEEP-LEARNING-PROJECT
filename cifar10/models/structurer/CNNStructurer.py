from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.optimizers import *
from cifar10.DatasetLoader import *
from tensorflow.keras import layers, activations,models,optimizers,metrics,datasets
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle as pk
from tensorflow.keras.regularizers import *
from tensorflow.keras.utils import plot_model
class CNNStructurer:

    def __init__(self):
        self.name = "mlp"
        self.nb_hidden_layers = 0
        self.layers_size = []
        self.layers_activation = 'relu'
        self.output_activation = 'softmax'
        self.use_dropout = False
        self.dropout_indexes = []
        self.dropout_value = 0.0
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.0
        self.l2_value = 0.0
        self.regulization_indexes = []
        self.loss = 'sparse_categorical_crossentropy'
        self.optimizer = Adam()
        self.metrics = ['sparse_categorical_accuracy']

if __name__ == "__main__":
    #importation des données:
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    #définition des noms des classes:
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    """
    #afficher les 25 premières lignes:
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
    """


#création du modèle:
    model = models.Sequential()
    #ajouter des couches de convolution:
    model.add(layers.Conv2D(32, (3, 3), activation='relu',kernel_regularizer=l1(0.02),input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='softplus'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Dropout(0.1))
    #applatir les données:
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    #afficher les détails :
    model.summary()
    #entrainer le model:
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=18,batch_size=5012,
                        validation_data=(test_images, test_labels),verbose=2)
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

#affichage du modele:
    plot_model(model, "../../test.png")