from random import randint, choice

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import *

from cifar10.models.structurer.MlpStructurer import MlpStructurer
from cifar10.models.structurer.UNetStructurer import UNetStructurer



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

def generateRandoMlpStruc(same_layers_depth=True, min_nb_layers=5, max_nb_layers=40, min_layer_depth=32, max_layer_depth=512):
    layers_activations = ['softmax', 'relu', 'softplus', 'selu']
    output_activations = ['softmax']
    metrics = [['sparse_categorical_accuracy']]
    losses = ['sparse_categorical_crossentropy']
    optimizers = [Adam()]
    regularisations = [True, False]
    possible_layers_sizes = []
    specific_size = min_layer_depth
    while specific_size <= max_layer_depth:
        possible_layers_sizes.append(specific_size)
        specific_size *= 2
    nb_layers = randint(min_nb_layers, max_nb_layers)
    layers_size = []
    if same_layers_depth:
        depth = choice(possible_layers_sizes)
        layers_size = [depth for i in range(nb_layers)]
    else:
        for i in range(nb_layers):
            layers_size.append(choice(possible_layers_sizes))
    use_dropout = choice(regularisations)
    use_l1l2 = choice(regularisations)
    use_l1l2_output = choice(regularisations)
    dropout_indexes = []
    dropout_value = 0.0
    if use_dropout:
        dropout_indexes_number = randint(1, nb_layers)
        dropout_value = randint(0, 4) / 10
        for j in range(dropout_indexes_number):
            dropout_indexes.append(randint(1, nb_layers))
    l1l2_indexes = []
    l1_value = 0.0
    l2_value = 0.0
    if use_l1l2:
        l1l2_indexes_number = randint(1, nb_layers)
        for j in range(l1l2_indexes_number):
            l1l2_indexes.append(randint(1, nb_layers))
        l1_value = randint(5, 100)/1000
        l2_value = randint(5, 100) / 1000

    struct = MlpStructurer()
    struct.nb_hidden_layers = nb_layers
    struct.layers_size = layers_size
    struct.layers_activation = choice(layers_activations)
    struct.output_activation = choice(output_activations)
    struct.use_dropout = use_dropout
    struct.dropout_indexes = dropout_indexes
    struct.dropout_value = dropout_value
    struct.use_l1l2_regularisation_hidden_layers = use_l1l2
    struct.use_l1l2_regularisation_output_layer = use_l1l2_output
    struct.l1_value = l1_value
    struct.l2_value = l2_value
    struct.regulization_indexes = l1l2_indexes
    struct.loss = choice(losses)
    struct.optimizer = choice(optimizers)
    struct.metrics = choice(metrics)

    return struct

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



################################################################## Begining of UNet Part ##########################################################################################

def create_unet(unet_struct: UNetStructurer):
    input_tensor = Input((32, 32, 3))

    layers_list = []
    maxpool_layers_list = []
    dropout_layers_list = []
    tensors_list = []
    tensors_to_connect_list_1 = []
    tensors_to_connect_list_2 = []

    for i in range(unet_struct.nb_Conv2D_layers):
        if unet_struct.use_l1l2_regularisation_hidden_layers and ((i + 1) in unet_struct.l1l2_regul_indexes):
            layer = Conv2D(filters=unet_struct.filter,
                           kernel_size=unet_struct.kernel_size,
                           activation=unet_struct.conv2D_activation,
                           kernel_regularizer=L1L2(unet_struct.l1_value, unet_struct.l2_value),
                           name=f"conv2d_l1l2_{i}",
                           padding=unet_struct.padding)
        else:
            layer = Conv2D(filters=unet_struct.filter,
                           kernel_size=unet_struct.kernel_size,
                           activation=unet_struct.conv2D_activation,
                           name=f"conv2d_{i}",
                           padding=unet_struct.padding)

        layers_list.append(layer)

    layers_list[0] = layers_list[0](input_tensor)

    if (unet_struct.nb_Conv2D_layers % 2 == 0):
        middle = int(unet_struct.nb_Conv2D_layers / 2)
    else :
        middle = int((unet_struct.nb_Conv2D_layers / 2) + 1)

    upsambled_layers_indexes = [(unet_struct.nb_Conv2D_layers - x) for x in unet_struct.MaxPooling2D_position if x <= middle]

    for j in range(middle - 1):
        tensors_to_connect_list_1.append(layers_list[j])
        if unet_struct.use_MaxPooling2D and (j+1 in unet_struct.MaxPooling2D_position):
            layers_list[j] = MaxPool2D(pool_size=(2, 2), name=f"maxpool_{j}")(layers_list[j])
        if unet_struct.use_dropout and (j+1 in unet_struct.dropout_indexes):
            layers_list[j] = Dropout(unet_struct.dropout_value, name=f"dropout_{j}")(layers_list[j])
        layers_list[j+1] = layers_list[j+1](layers_list[j])

    for j in range(middle, unet_struct.nb_Conv2D_layers):
        if j in upsambled_layers_indexes:
            layers_list[j - 1] = UpSampling2D(name=f"upsample_{j}")(layers_list[j - 1])
        tensors_to_connect_2 = layers_list[j-1]
        tensors_to_connect_1 = tensors_to_connect_list_1.pop()
        # if unet_struct.use_MaxPooling2D and (j in unet_struct.MaxPooling2D_position):
        #     layers_list[j - 1] = MaxPool2D(pool_size=(2, 2), name=f"maxpool_{j}")(layers_list[j - 1])
        # if unet_struct.use_dropout and (j in unet_struct.dropout_indexes):
        #     layers_list[j - 1] = Dropout(unet_struct.dropout_value, name=f"dropout_{j}")(layers_list[j - 1])
        avg_tensor = Average()([tensors_to_connect_2, tensors_to_connect_1])
        layers_list[j] = layers_list[j](avg_tensor)

    flatten_tensor = Flatten(name="flatten")(layers_list[-1])
    output_tensor = Dense(10, activation=unet_struct.output_activation, name="output_dense")(flatten_tensor)

    model = Model(input_tensor, output_tensor)

    model.compile(loss=unet_struct.loss, optimizer=unet_struct.optimizer, metrics=unet_struct.metrics)

    return model







################################################################## End of UNet Part ##########################################################################################

#################################################################### Commons Part ###########################################################################################
def getRandomModelID():
    uid = randint(0, 10000000)
    return "{:07d}".format(uid)
