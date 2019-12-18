from cifar10.DatasetLoader import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *



if __name__ == "__main__":
    dataset_dir = "cifar-10-batches-py"
    
    label_names, num_cases_per_batch, num_vis = readMetaData(dataset_dir)
    x_train, y_train, train_file_names = readTrainDataset(dataset_dir)
    x_test, y_test, test_file_names = readTestDataset(dataset_dir)

    model = Sequential()
    model.add(Dense(128, activation=relu, input_dim=3072))
    model.add(Dense(128, activation=relu))
    model.add(Dense(10, activation=softmax))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=[sparse_categorical_accuracy])

    model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=2000,
          batch_size=4096
          )

