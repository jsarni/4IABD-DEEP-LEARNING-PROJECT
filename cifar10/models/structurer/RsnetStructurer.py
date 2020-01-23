from tensorflow.keras.optimizers import *




class RsnetStructurer:

    def __init__(self):
        self.name = "resnet34"
        self.kernel_size = (3, 3)
        self.filters = 32
        self.batch_size = 32
        self.input_shape = (32, 32, 3)
        self.nb_hidden_layers = 2
        self.layers_activation = 'relu'
        self.output_activation = 'softmax'
        self.padding = "same"
        self.nb_skip = 2
        self.use_skip = True
        self.use_dropout = False
        self.dropout_indexes = [2, 4]
        self.dropout_value = 0.2
        self.use_l1l2_regularisation_hidden_layers = False
        self.use_l1l2_regularisation_output_layer = False
        self.l1_value = 0.004
        self.l2_value = 0.002
        self.regulization_indexes = [5, 1]
        self.loss = 'categorical_crossentropy'
        self.optimizer = Adam()
        self.metrics = ['categorical_accuracy']


