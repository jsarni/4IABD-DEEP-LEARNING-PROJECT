from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from cifar10.models.LSTM import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras.regularizers import L1L2
if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    train_data = train_data.reshape((50000, 32, 96))
    val_data = val_data.reshape((10000, 32, 96))
    epochs = [100]

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 5
        struct.recurrent_dropout_value = 0.5
        struct.units = 128
        struct.l2_value = 0.02
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256,
                    save_image=True)
        clear_session()

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 5
        struct.recurrent_dropout_value = 0.5
        struct.units = 128
        struct.l2_value = 0.005
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256,
                    save_image=True)
        clear_session()

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 5
        struct.recurrent_dropout_value = 0.5
        struct.units = 128
        struct.l2_value = 0.005
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        struct.output_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256,
                    save_image=True)
        clear_session()

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 7
        struct.recurrent_dropout_value = 0.5
        struct.units = 128
        struct.l2_value = 0.005
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=256,
                    save_image=True)
        clear_session()

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 9
        struct.recurrent_dropout_value = 0.4
        struct.units = 128
        struct.l2_value = 0.001
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=128,
                    save_image=True)
        clear_session()

    for i in range(1):
        struct = LstmStructurer()
        struct.nb_layers = 9
        struct.recurrent_dropout_value = 0.4
        struct.units = 128
        struct.l2_value = 0.01
        struct.recurrent_regularizer = L1L2(l2=struct.l2_value)
        model = [create_lstm(struct)]
        desc = [getLstmStructAsString(struct)]
        print(desc[0])
        test_models('lstm_128', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=128,
                    save_image=True)
        clear_session()




