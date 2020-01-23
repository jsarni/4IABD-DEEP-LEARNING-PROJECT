from cifar10.models.Models import *
from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import clear_session


if __name__ == '__main__':

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    epochs = [500]

    csv_file = ".\\trained_models\\mlp\\best_mlp_without_regularisation.csv"
    structs = generateStructsFromCSV(csv_file)
    for struct in structs:
        for j in range(3):
            struct.use_dropout = True
            dropout_indexes = []
            dropout_indexes_number = randint(1, struct.nb_hidden_layers)
            dropout_value = randint(1, 3) / 10
            for j in range(dropout_indexes_number):
                dropout_indexes.append(randint(1, struct.nb_hidden_layers))

            model = [create_custom_mlp(struct)]
            desc = [getMlpStructAsString(struct)]

            print(desc[0])
            test_models('mlp_best_models_with_dropout', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=8192)
            clear_session() #clear tf GPU memory

