from tensorflow.keras.datasets import *

from cifar10.models.Models import *

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__, str(__package__)))


def train_model_mlp(model, train_ds, train_labels, test_ds, test_labels, epochs_p=200, batch_size_p=4096):
    return model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=epochs_p,
                     batch_size=batch_size_p)


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    mlp_struct = MlpStructurer()

    mlp_struct.nb_hidden_layers = 2
    mlp_struct.layers_size = [10, 10]

    m = create_custom_mlp(mlp_struct)
    m.summary()

    # train_model_mlp(m, train_images, train_labels, test_images, test_labels, 1)
