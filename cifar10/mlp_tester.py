from cifar10.models.ModelTester import *
from cifar10.models.Models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import cifar10


if __name__ == "__main__":

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    nb_hidden_layers_list = [2, 3]
    layers_size_list = [[10, 10], [5, 5, 5]]
    layers_activation_list = ['relu', 'relu']
    output_activation_list = ['softmax', 'softmax']
    use_dropout_list = [False, False]
    dropout_indexes_list = [[], []]
    dropout_value_list = [0, 0]
    use_l1l2_regularisation_hidden_layers_list = [False, False]
    use_l1l2_regularisation_output_layer_list = [False, False]
    l1_value_list = [0, 0]
    l2_value_list = [0, 0]
    regulization_indexes_list = [[], []]
    loss_list = ['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy']
    optimizer_list = [Adam(), Adam()]
    metrics_list = [['sparse_categorical_accuracy'], ['sparse_categorical_accuracy']]

    epochs = [5]

    models, descriptions = generateMlpModels(nb_hidden_layers_list,
                                      layers_size_list,
                                      layers_activation_list,
                                      output_activation_list,
                                      use_dropout_list,
                                      dropout_indexes_list,
                                      dropout_value_list,
                                      use_l1l2_regularisation_hidden_layers_list,
                                      use_l1l2_regularisation_output_layer_list,
                                      l1_value_list,
                                      l2_value_list,
                                      regulization_indexes_list,
                                      loss_list,
                                      optimizer_list,
                                      metrics_list
                                      )

    nb_tested_models = test_models('mlp', models, descriptions, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=4096)

    print("END : Number of tested models = " + str(nb_tested_models))