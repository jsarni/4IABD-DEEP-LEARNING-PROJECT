from cifar10.models.ModelTester import *
from tensorflow.keras.datasets import cifar10
from cifar10.models.Models import *
import tensorflow as tf



(train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
train_labels =tf.keras.utils.to_categorical(train_labels, 10)
val_labels = tf.keras.utils.to_categorical(val_labels, 10)
epochs = [30]
for i in range(10):
    struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
    model = [create_model_resenet34(struct)]
    desc = [getResetStructAsString(struct)]
    test_models('rsnet_3_6_32', model, desc, train_data, train_labels, val_data, val_labels,
                epochs_p=epochs,
                batch_size_p=struct.batch_size)

del model
del desc