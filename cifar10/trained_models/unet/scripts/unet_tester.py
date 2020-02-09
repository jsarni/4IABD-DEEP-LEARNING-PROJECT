
from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from cifar10.models.UNet import *

from tensorflow.keras.backend import clear_session

if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    epochs = [40]

    for i in range(3):
        struct = generateRandoUNetStruc(min_nb_layers=7,max_nb_layers=7)
        struct.use_MaxPooling2D = True
        struct.MaxPooling2D_position = [1, 2, 3]
        model = [create_unet(struct)]
        desc = [getUNetStructAsString(struct)]
        print(desc[0])
        test_models('model_3', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256, save_image=True, save_model=True)
        clear_session()
