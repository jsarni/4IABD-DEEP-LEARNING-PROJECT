from cifar10.models.ModelTester import test_models
from tensorflow.keras.datasets import cifar10
from cifar10.models.LSTM import *
from tensorflow.keras.backend import clear_session

if __name__ == "__main__":
    (train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
    train_data = train_data.reshape((50000, 32, 96))
    val_data = val_data.reshape((10000, 32, 96))
    epochs = [10]

    struct = LstmStructurer()

    model = [create_lstm(struct)]
    desc = [getLstmStructAsString(struct)]
    print(desc[0])
    test_models('lstm', model, desc, train_data, train_labels, val_data, val_labels, epochs_p=epochs, batch_size_p=1024,
                save_image=True)
    clear_session()
