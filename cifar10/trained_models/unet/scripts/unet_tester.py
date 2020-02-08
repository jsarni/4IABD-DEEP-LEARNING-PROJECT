
from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from cifar10.models.UNet import *

from tensorflow.keras.backend import clear_session

if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    epochs = [40]

    for i in range(3):
        struct = generateRandoUNetStruc(min_nb_layers=5,max_nb_layers=5)
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.l1l2_regul_indexes = [1, 2, 3, 4, 5]
        struct.l2_value = 0.5
        model = [create_unet(struct)]
        desc = [getUNetStructAsString(struct)]
        print(desc[0])
        test_models('unet_l2', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256, save_image=True)
        clear_session()

    for i in range(3):
        struct = generateRandoUNetStruc(min_nb_layers=7,max_nb_layers=7)
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.l1l2_regul_indexes = [1, 2, 3, 4, 5, 6, 7]
        struct.l2_value = 0.5
        model = [create_unet(struct)]
        desc = [getUNetStructAsString(struct)]
        print(desc[0])
        test_models('unet_l2', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256, save_image=True)
        clear_session()

    for i in range(3):
        struct = generateRandoUNetStruc(min_nb_layers=9,max_nb_layers=9)
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.l1l2_regul_indexes = [1, 2, 3, 4, 5, 6, 7, 9]
        struct.l2_value = 0.5
        model = [create_unet(struct)]
        desc = [getUNetStructAsString(struct)]
        print(desc[0])
        test_models('unet_l2', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256, save_image=True)
        clear_session()