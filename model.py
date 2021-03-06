import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import initializers

print(tf.__version__)

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


def KeypointModel(input_size=128, numclass=196):

    mobile_model = MobileNet(
        weights="imagenet",
        alpha=0.5,
        input_tensor=layers.Input(shape=(input_size, input_size, 3),
                                  name='feature'),
        include_top=False)

    for layer in mobile_model.layers:
        layer.trainable = True

    x = mobile_model.output
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(numclass,
                          name='predictions',
                          kernel_initializer=initializers.he_normal(42))(x)
    model = Model(inputs=mobile_model.input,
                  outputs=output, name='facekeypoint')

    return model
