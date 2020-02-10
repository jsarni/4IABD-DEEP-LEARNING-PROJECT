from cifar10.models.ModelTester import *
from cifar10.models.MLP import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.backend import clear_session


if __name__ == "__main__":

    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()

    epochs = [200]
    for i in range(3):
        struct = MlpStructurer()
        struct.nb_hidden_layers = 11
        struct.layers_size = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
        struct.l2_value = 0.001
        struct.use_l1l2_regularisation_hidden_layers = True
        struct.regulization_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        struct.layers_activation = 'softplus'
        model = [create_custom_mlp(struct)]
        desc = [getMlpStructAsString(struct)]
        print(desc[0])
        test_models('models_1', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=4096,
                    save_image=True, save_model=True)
        clear_session()