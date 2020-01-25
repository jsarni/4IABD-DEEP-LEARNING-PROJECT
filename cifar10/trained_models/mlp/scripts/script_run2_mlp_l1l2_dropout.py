from cifar10.models.Models import *
from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import clear_session


if __name__ == '__main__':

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    epochs = [500]

    csv_file = "../historique_tests/best_mlp_without_regularisation.csv"
    structs = generateStructsFromCSV(csv_file)
    struct = structs[5] # meilleur model (à 11 couches)
    # for struct in structs:
    for j in range(3):
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.regulization_indexes = [x+1 for x in range(struct.nb_hidden_layers)]
        struct.l2_value = 0.001
        struct.l1_value = 0.0

        struct.use_dropout = True
        struct.dropout_value = 0.1 * (j+1) # Pour avoir des dropouts à 0.1, puis 0.2, puis 0.3
        struct.dropout_indexes = [x+1 for x in range(struct.nb_hidden_layers)]
        model = [create_custom_mlp(struct)]
        desc = [getMlpStructAsString(struct)]

        print(desc[0])
        test_models('mlp_best_models_with_l2_and_dropout', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)
        clear_session() #clear tf GPU memory

    for j in range(3):
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.regulization_indexes = [x+1 for x in range(struct.nb_hidden_layers)]
        struct.l2_value = 0.002
        struct.l1_value = 0.0

        struct.use_dropout = True
        struct.dropout_value = 0.1 * (j+1) # Pour avoir des dropouts à 0.1, puis 0.2, puis 0.3
        struct.dropout_indexes = [x+1 for x in range(struct.nb_hidden_layers)]
        model = [create_custom_mlp(struct)]
        desc = [getMlpStructAsString(struct)]

        print(desc[0])
        test_models('mlp_best_models_with_l2_and_dropout', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)
        clear_session() #clear tf GPU memory