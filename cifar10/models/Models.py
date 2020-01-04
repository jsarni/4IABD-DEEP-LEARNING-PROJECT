from random import randint

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *

from cifar10.models.structurer.MlpStructurer import MlpStructurer


################################################################## Beginin of MLP Part ##########################################################################################

def create_custom_mlp(mlp_struct: MlpStructurer):
    assert mlp_struct.nb_hidden_layers == len(
        mlp_struct.layers_size), "MlpStructurerError: MLP number of layers (nb_hidden_layers) is different of the total layers sizes number (layer_size) "

    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))
    for i in range(len(mlp_struct.layers_size)):

        # Hidden layers L1L2 regularisation
        if mlp_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in mlp_struct.regulization_indexes):
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation,
                            kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value),
                            name=f"dense_l1l2_{i}"
                            )
                      )
        else:
            model.add(Dense(mlp_struct.layers_size[i],
                            activation=mlp_struct.layers_activation,
                            name=f"dense_{i}"
                            )
                      )
        if mlp_struct.use_dropout and ((i + 1) in mlp_struct.dropout_indexes):
            model.add(Dropout(mlp_struct.dropout_value, name=f"dropout_{i}"))

    # Output L1L2 regularisation
    if mlp_struct.use_l1l2_regularisation_output_layer:
        model.add(Dense(10,
                        activation=mlp_struct.output_activation,
                        kernel_regularizer=L1L2(l1=mlp_struct.l1_value, l2=mlp_struct.l2_value),
                        name="output_l1l2"
                        )
                  )
    else:
        model.add(Dense(10,
                        activation=mlp_struct.output_activation,
                        name="output"
                        )
                  )

    model.compile(loss=mlp_struct.loss, optimizer=mlp_struct.optimizer, metrics=mlp_struct.metrics)

    return model


def generateMlpModels(nb_hidden_layers_list: list,
                          layers_size_list: list,
                          layers_activation_list: list,
                          output_activation_list: list,
                          use_dropout_list: list,
                          dropout_indexes_list: list,
                          dropout_value_list: list,
                          use_l1l2_regularisation_hidden_layers_list: list,
                          use_l1l2_regularisation_output_layer_list: list,
                          l1_value_list: list,
                          l2_value_list: list,
                          regulization_indexes_list: list,
                          loss_list: list,
                          optimizer_list: list,
                          metrics_list: list):
    assert len(nb_hidden_layers_list) == len(layers_size_list)
    assert len(nb_hidden_layers_list) == len(layers_activation_list)
    assert len(nb_hidden_layers_list) == len(output_activation_list)
    assert len(nb_hidden_layers_list) == len(use_dropout_list)
    assert len(nb_hidden_layers_list) == len(dropout_indexes_list)
    assert len(nb_hidden_layers_list) == len(dropout_value_list)
    assert len(nb_hidden_layers_list) == len(use_l1l2_regularisation_hidden_layers_list)
    assert len(nb_hidden_layers_list) == len(use_l1l2_regularisation_output_layer_list)
    assert len(nb_hidden_layers_list) == len(l1_value_list)
    assert len(nb_hidden_layers_list) == len(l2_value_list)
    assert len(nb_hidden_layers_list) == len(regulization_indexes_list)
    assert len(nb_hidden_layers_list) == len(loss_list)
    assert len(nb_hidden_layers_list) == len(optimizer_list)
    assert len(nb_hidden_layers_list) == len(metrics_list)

    mlp_models = []
    mlp_descriptions = []

    current_structure = MlpStructurer()

    for i in range(len(nb_hidden_layers_list)):
        current_structure.nb_hidden_layers = nb_hidden_layers_list[i]
        current_structure.layers_size = layers_size_list[i]
        current_structure.layers_activation = layers_activation_list[i]
        current_structure.output_activation = output_activation_list[i]
        current_structure.use_dropout = use_dropout_list[i]
        current_structure.dropout_indexes = dropout_indexes_list[i]
        current_structure.dropout_value = dropout_value_list[i]
        current_structure.use_l1l2_regularisation_hidden_layers = use_l1l2_regularisation_hidden_layers_list[i]
        current_structure.use_l1l2_regularisation_output_layer = use_l1l2_regularisation_output_layer_list[i]
        current_structure.l1_value = l1_value_list[i]
        current_structure.l2_value = l2_value_list[i]
        current_structure.regulization_indexes = regulization_indexes_list[i]
        current_structure.loss = loss_list[i]
        current_structure.optimizer = optimizer_list[i]
        current_structure.metrics = metrics_list[i]

        mlp_models.append(create_custom_mlp(current_structure))
        mlp_descriptions.append(getMlpStructAsString(current_structure))

    return mlp_models, mlp_descriptions

def getMlpStructAsString(mlp_structurer):
    return "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(mlp_structurer.nb_hidden_layers,
                                                                    " ".join([str(i) for i in mlp_structurer.layers_size]),
                                                                    mlp_structurer.layers_activation,
                                                                    mlp_structurer.output_activation,
                                                                    mlp_structurer.use_dropout,
                                                                    " ".join([str(i) for i in mlp_structurer.dropout_indexes]),
                                                                    mlp_structurer.dropout_value,
                                                                    mlp_structurer.use_l1l2_regularisation_hidden_layers,
                                                                    mlp_structurer.use_l1l2_regularisation_output_layer,
                                                                    mlp_structurer.l1_value,
                                                                    mlp_structurer.l2_value,
                                                                    " ".join([str(i) for i in mlp_structurer.regulization_indexes]),
                                                                    mlp_structurer.loss,
                                                                    mlp_structurer.optimizer.__class__.__name__,
                                                                    " ".join(mlp_structurer.metrics)
                                                                    )


################################################################## End of MLP Part ##########################################################################################


################################################################## Commons Part ##########################################################################################
def getRandomModelID():
    uid = randint(0, 10000000)
    return "{:07d}".format(uid)
