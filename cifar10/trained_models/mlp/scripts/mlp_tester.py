from cifar10.models.ModelTester import *
from cifar10.models.MLP import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import cifar10


if __name__ == "__main__":

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    epochs = [400]
    # nb_hidden_layers_list = [10]
    # layers_size_list = [[64, 64, 64, 64, 64, 64, 64, 64, 64, 64]]
    # layers_activation_list = ['softplus']
    # output_activation_list = ['softmax']
    # use_dropout_list = [False]
    # dropout_indexes_list = [[]]
    # dropout_value_list = [0.2]
    # use_l1l2_regularisation_hidden_layers_list = [True]
    # use_l1l2_regularisation_output_layer_list = [True]
    # l1_value_list = [0.01]
    # l2_value_list = [0.01]
    # regulization_indexes_list = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    # loss_list = ['sparse_categorical_crossentropy']
    # optimizer_list = [Adam()]
    # metrics_list = [['sparse_categorical_accuracy']]


    # Manual model testing
    # models, descriptions = generateMlpModels(nb_hidden_layers_list,
    #                                   layers_size_list,
    #                                   layers_activation_list,
    #                                   output_activation_list,
    #                                   use_dropout_list,
    #                                   dropout_indexes_list,
    #                                   dropout_value_list,
    #                                   use_l1l2_regularisation_hidden_layers_list,
    #                                   use_l1l2_regularisation_output_layer_list,
    #                                   l1_value_list,
    #                                   l2_value_list,
    #                                   regulization_indexes_list,
    #                                   loss_list,
    #                                   optimizer_list,
    #                                   metrics_list
    #                                   )
    #
    # nb_tested_models = test_models('mlp', models, descriptions, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)
    #
    # print("END : Number of tested models = " + str(nb_tested_models))

    # Automatic model testing

    while True:
        struct = generateRandoMlpStruc(same_layers_depth=True, min_nb_layers=5, max_nb_layers=30, min_layer_depth=64, max_layer_depth=512)
        model = [create_custom_mlp(struct)]
        desc = [getMlpStructAsString(struct)]
        print(desc[0])
        test_models('mlp', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)

