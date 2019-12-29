from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from .MlpStructurer import MlpStructurer
from tensorflow.keras.regularizers import *
from random import randint

def create_custom_mlp(mlp_struct: MlpStructurer):

    assert mlp_struct.nb_hidden_layers == len(mlp_struct.layers_size), "MlpStructurerError: MLP number of layers (nb_hidden_layers) is different of the total layers sizes number (layer_size) "

    model = Sequential()
    model.add(Flatten(input_shape=(32,32,3)))
    for i in range(len(mlp_struct.layers_size)):

        # Hidden layers L1L2 regularisation
        if mlp_struct.use_l1l2_regularisation_hidden_layers and ((i+1) in mlp_struct.regul_kernel_indexes):
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation,
                            kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value)
                            )
                      )
        else :
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation))
        if mlp_struct.use_dropout and ((i+1) in mlp_struct.regul_kernel_indexes):
            model.add(Dropout(mlp_struct.dropout_value))

    # Output L1L2 regularisation
    if mlp_struct.use_l1l2_regularisation_output_layer:
        model.add(Dense(10,
                        activation=mlp_struct.output_activation,
                        kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value)
                        )
                  )
    else:
        model.add(Dense(10,
                        activation=mlp_struct.output_activation))

    model.compile(loss=mlp_struct.loss, optimizer=mlp_struct.optimizer, metrics=mlp_struct.metrics)

    return model

def getModelLogNameFromMlpStructurer(mlp_structurer: MlpStructurer):
    uid = randint(0, 100)
    return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log".format(mlp_structurer.name,
                                                                    mlp_structurer.nb_hidden_layers,
                                                                    "_".join(mlp_structurer.layers_size),
                                                                    mlp_structurer.layers_activation,
                                                                    mlp_structurer.output_activation,
                                                                    mlp_structurer.use_dropout,
                                                                    "_".join(mlp_structurer.dropout_indexes),
                                                                    mlp_structurer.dropout_value,
                                                                    mlp_structurer.use_l1l2_regularisation_hidden_layers,
                                                                    mlp_structurer.use_l1l2_regularisation_output_layer,
                                                                    mlp_structurer.l1_value,
                                                                    mlp_structurer.l2_value,
                                                                    "_".join(mlp_structurer.regul_kernel_indexes),
                                                                    mlp_structurer.loss,
                                                                    mlp_structurer.optimizer.__class__.__name__,
                                                                    "_".join(mlp_structurer.metrics),
                                                                    uid
                                                                    )
