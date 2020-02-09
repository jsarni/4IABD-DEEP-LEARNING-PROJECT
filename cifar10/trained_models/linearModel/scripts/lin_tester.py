from tensorflow.keras.datasets import *

from cifar10.models.LIN import create_lin_model, getLinStructAsString
from cifar10.models.ModelTester import test_models
from cifar10.models.structurer.LinearStructurer import LinearStructurer


if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    epochs = [100]

    struct = LinearStructurer()
    #struct.loss ='binary_crossentropy'
    desc = [getLinStructAsString(struct)]
    linear_model = [create_lin_model(struct)]
    test_models('lin', linear_model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=1024)
