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




################################################################# Resnet PArt###############################################################################################

def create_model_resenet34(RsnetStruct: RsnetStructurer):


    input_tensor = Input((32, 32, 3))
    last_output_tensor = input_tensor

    antipen_output_tensor = None
    nb_skipped = 0
    for i in range(RsnetStruct.nb_hidden_layers):
        if RsnetStruct.use_skip and antipen_output_tensor is not None:
            if nb_skipped == RsnetStruct.nb_skip:
                add_tensor = Add()([antipen_output_tensor, last_output_tensor])
                antipen_output_tensor = add_tensor
                last_output_tensor = add_tensor
                nb_skipped = 0

                # Hidden layers L1L2 regularisation
            if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding, activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value, l2=RsnetStruct.l2_value), input_shape=(32, 32, 3),name=f"conv2d__L1L2_{i}")(last_output_tensor)
            else:
                last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3),name=f"conv2d_{i}")(last_output_tensor)
            # Use dropout
            if (RsnetStruct.use_dropout and (i + 1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)




            nb_skipped += 1

        else:
            if( RsnetStruct.use_skip and i==0):

                # Hidden layers L1L2 regularisation
                if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,  l2=RsnetStruct.l2_value),input_shape=(32, 32, 3),  name=f"conv2d__L1L2_{i}")(last_output_tensor)
                else:
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3),name=f"conv2d_{i}")(last_output_tensor)
                # Use dropout
                if(RsnetStruct.use_dropout and (i+1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)

                antipen_output_tensor= Dense(32, activation=RsnetStruct.layers_activation, name=f"dense_{i}")(input_tensor)


            else:
                antipen_output_tensor = input_tensor
                # Hidden layers L1L2 regularisation
                if RsnetStruct.use_l1l2_regularisation_hidden_layers and ((i + 1) in RsnetStruct.regulization_indexes):
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,activation=RsnetStruct.layers_activation,
                                            kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,l2=RsnetStruct.l2_value),input_shape=(32, 32, 3),name=f"conv2d__L1L2_{i}")(last_output_tensor)
                else:
                    last_output_tensor = Conv2D(RsnetStruct.filters, RsnetStruct.kernel_size,padding=RsnetStruct.padding,
                                                activation=RsnetStruct.layers_activation, input_shape=(32, 32, 3), name=f"conv2d_{i}")(last_output_tensor)
                # Use dropout
                if (RsnetStruct.use_dropout and (i + 1) in RsnetStruct.dropout_indexes):
                    last_output_tensor = Dropout(RsnetStruct.dropout_value, name=f"dropout_{i}")(last_output_tensor)
            nb_skipped += 1

    if RsnetStruct.use_skip and RsnetStruct.nb_hidden_layers % RsnetStruct.nb_skip == 0:
        last_output_tensor = Add()([antipen_output_tensor, last_output_tensor])

    flattended_last_tensor = Flatten(name='flatten')(last_output_tensor)
    if (RsnetStruct.use_l1l2_regularisation_output_layer):
        output_tensor = Dense(10,RsnetStruct.output_activation, kernel_regularizer=L1L2(l1=RsnetStruct.l1_value,l2=RsnetStruct.l2_value),name='dense_output_L1L2')(flattended_last_tensor)
    else:
        output_tensor = Dense(10, RsnetStruct.output_activation, name='dense_output')(flattended_last_tensor)
    model = Model(input_tensor, output_tensor)

    model.compile(loss=RsnetStruct.loss,
                  optimizer=RsnetStruct.optimizer,
                  metrics=RsnetStruct.metrics)

    return model






######################################################### End Resnet #######################################################################################


################################################################## Commons Part ##########################################################################################
def getRandomModelID():
    uid = randint(0, 10000000)
    return "{:07d}".format(uid)
