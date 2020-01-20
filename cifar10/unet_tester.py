from cifar10.models.Models import *
from cifar10.models.structurer.UNetStructurer import UNetStructurer
from tensorflow.keras.utils import plot_model

if __name__ == "__main__":

    struct = UNetStructurer()

    struct.nb_Conv2D_layers = 7
    struct.use_MaxPooling2D = True
    struct.MaxPooling2D_position = [2]

    model = create_unet(struct)
    plot_model(model, "unet_maxpool_upsampled_7.png")
    model.summary()