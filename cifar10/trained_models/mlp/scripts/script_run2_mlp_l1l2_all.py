from cifar10.models.MLP import *
from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import clear_session


if __name__ == '__main__':

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    epochs = [500]

    csv_file = "../historique_tests/best_mlp_without_regularisation.csv"
    structs = generateStructsFromCSV(csv_file)
    struct = structs[5]
    # for struct in structs:
    for j in range(1):
        struct.use_l1l2_regularisation_hidden_layers = True
        # struct.use_l1l2_regularisation_output_layer = True
        struct.regulization_indexes = [x for x in range(struct.nb_hidden_layers)]
        # dropout_indexes_number = randint(1, struct.nb_hidden_layers)
        struct.l2_value = 0.001
        struct.l1_value = 0.0
        # for j in range(dropout_indexes_number):
        #     dropout_indexes.append(randint(1, struct.nb_hidden_layers))

        model = [create_custom_mlp(struct)]
        desc = [getMlpStructAsString(struct)]

        print(desc[0])
        test_models('mlp_best_models_with_l2', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)
        clear_session() #clear tf GPU memory
