from tensorflow.keras.datasets import *
from tensorflow.keras.optimizers import *

from cifar10.models.Models import *


def train_model_mlp(model, train_ds, train_labels, test_ds, test_labels, epochs_p=200, batch_size_p=4096):
    return model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=epochs_p,
                     batch_size=batch_size_p)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    mlp_struct = MlpStructurer()

    mlp_struct.nb_hidden_layers = 2
    mlp_struct.layers_size = [10, 10]
    mlp_struct.use_dropout = True
    mlp_struct.dropout_indexes = [1]
    mlp_struct.use_l1l2_regularisation_hidden_layers = True
    mlp_struct.regulization_indexes = [1]
    mlp_struct.use_l1l2_regularisation_output_layer = True
    mlp_struct.optimizer = SGD()
    mlp_struct.metrics = ['mean_squared_error']

    m = create_custom_mlp(mlp_struct)
    m.summary()