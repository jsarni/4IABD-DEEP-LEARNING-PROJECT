from random import randint
from cifar10.models.structurer.MlpStructurer import MlpStructurer

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import *






def create_custom_mlp(mlp_struct: MlpStructurer):
    assert mlp_struct.nb_hidden_layers == len(mlp_struct.layers_size), "MlpStructurerError: MLP number of layers (nb_hidden_layers) is different of the total layers sizes number (layer_size) "

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


def getModelLogNameFromMlpStructurer(mlp_structurer: MlpStructurer):
    uid = randint(0, 1000)
    return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.log".format(mlp_structurer.name,
                                                                           mlp_structurer.nb_hidden_layers,
                                                                           "_".join([str(i) for i in mlp_structurer.layers_size]),
                                                                           mlp_structurer.layers_activation,
                                                                           mlp_structurer.output_activation,
                                                                           mlp_structurer.use_dropout,
                                                                           "_".join([str(i) for i in mlp_structurer.dropout_indexes]),
                                                                           mlp_structurer.dropout_value,
                                                                           mlp_structurer.use_l1l2_regularisation_hidden_layers,
                                                                           mlp_structurer.use_l1l2_regularisation_output_layer,
                                                                           mlp_structurer.l1_value,
                                                                           mlp_structurer.l2_value,
                                                                           "_".join([str(i) for i in mlp_structurer.regulization_indexes]),
                                                                           mlp_structurer.loss,
                                                                           mlp_structurer.optimizer.__class__.__name__,
                                                                           "_".join(mlp_structurer.metrics),
                                                                           uid
                                                                           )



def create_model_resnet34(num_hidden: int = 12, use_skip_connections: bool = False):

    input_tensor = Input((28, 28, 1))
    last_output_tensor = input_tensor
    antipen_output_tensor = None
    nb_skipped = 0
    for i in range(num_hidden):
        if use_skip_connections and antipen_output_tensor is not None:
                if nb_skipped == 2:
                    add_tensor = Add()([antipen_output_tensor, last_output_tensor])
                    antipen_output_tensor = add_tensor
                    last_output_tensor = add_tensor
                    nb_skipped = 0
                last_output_tensor = Conv2D(8, (28, 28), padding='same', activation='relu', input_shape=(28, 28, 1), name=f"conv2d_{i}")(last_output_tensor)
                nb_skipped += 1

        else:
            antipen_output_tensor =input_tensor
            last_output_tensor = Conv2D(8, (28, 28), padding='same', activation='relu', input_shape=(28, 28, 1), name=f"conv2d_{i}")(last_output_tensor)
            nb_skipped +=1

    if use_skip_connections and num_hidden % 2 == 0:
        last_output_tensor = Add()([antipen_output_tensor, last_output_tensor])

    flattended_last_tensor = Flatten(name='flatten')(last_output_tensor)

    output_tensor = Dense(10, activation=softmax, name='dense_output')(flattended_last_tensor)

    model = Model(input_tensor, output_tensor)

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(),
                  metrics=[sparse_categorical_accuracy])

    return model
