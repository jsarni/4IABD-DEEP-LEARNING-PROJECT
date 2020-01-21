from tensorflow.keras.utils import plot_model

from cifar10.models.ModelTester import test_models
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import cifar10
from cifar10.models.Models import *
from cifar10.models.structurer.UNetStructurer import UNetStructurer

if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    epochs = [10]

    while True:
        struct = generateRandoUNetStruc(max_nb_layers=10)
        model = [create_unet(struct)]
        desc = [getUNetStructAsString(struct)]
        print(desc[0])
        test_models('unet', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256)