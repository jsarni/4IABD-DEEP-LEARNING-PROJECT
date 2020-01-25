from cifar10.models.ModelTester import *
from tensorflow.keras.backend import clear_session
from tensorflow.keras.datasets import cifar10
from cifar10.models.Models import *
import tensorflow as tf



(train_data, train_labels), (val_data, val_labels) = cifar10.load_data()
train_labels =tf.keras.utils.to_categorical(train_labels, 10)
val_labels = tf.keras.utils.to_categorical(val_labels, 10)
epochs = [50]

# ############################# 3 à 6 Conv2D sans droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_3_6_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

############################# 6 à 8 Conv2D sans droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=6, max_nb_layers=8)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_6_8_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#
# del model
# del desc

# ############################# 8 à 12 Conv2D sans droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=8, max_nb_layers=10)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_8_12_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#
# del model
# del desc

# ############################# 12 à 15 Conv2D sans droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=8, max_nb_layers=10)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_12_15_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#
# del model
# del desc






# ############################# 3 à 6 Conv2D avec droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_dropout_3_6_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 6 à 8 Conv2D avec droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_dropout_6_8_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 8 à 12 Conv2D avec droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_dropout_8_12_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 12 à 15 Conv2D avec droupout et sans regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=False, use_l1l2_output=False, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_dropout_12_15_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc







# ############################# 3 à 6 Conv2D sans droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_3_6_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 6 à 8 Conv2D sans droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_6_8_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 8 à 12 Conv2D sans droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_8_12_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 12 à 15 Conv2D sans droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=False, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_12_15_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc




# ############################# 3 à 6 Conv2D avec droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_dropout_3_6_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 6 à 8 Conv2D avec droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_dropout_6_8_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 8 à 12 Conv2D avec droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_dropout_8_12_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc

# ############################# 12 à 15 Conv2D avec droupout et avec regul ###############################################
# for i in range(10):
#     struct = generateRandomRsnetStruc(use_maxpool=False, use_l1l2_hidden=True, use_l1l2_output=True, use_dropout=True, use_skip = True ,nb_skip =2, min_nb_layers=3, max_nb_layers=6)
#     model = [create_model_resenet34(struct)]
#     desc = [getResetStructAsString(struct)]
#     test_models('rsnet_regul_drop_out_12_15_32', model, desc, train_data, train_labels, val_data, val_labels,
#                 epochs_p=epochs,
#                 batch_size_p=struct.batch_size)
#     clear_session()
# del model
# del desc