from cifar10.models.ModelTester import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import cifar10
from cifar10.models.Rsnet import *
import tensorflow as tf


(train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
train_labels =tf.keras.utils.to_categorical(train_labels, 10)
val_labels = tf.keras.utils.to_categorical(val_labels, 10)
epochs = [50]



    
#### regul  3 to 4 ####
for i in range(3):
    clear_session()
    struct =RsnetStructurer()
        # generateRandomRsnetStruc(use_maxpool=True, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=4)
    model = [create_model_resenet34(struct)]
    desc = [getResetStructAsString(struct)]
    test_models('My_Rsnet', model, desc, train_data, train_labels, val_data, val_labels,save_image=True,save_model=True,
                epochs_p=epochs,
                batch_size_p=struct.batch_size)

    clear_session()
del model
del desc
