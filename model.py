import numpy as np
import tensorflow as tf
from tensorflow import keras as k

np.random.seed(42)
tf.random.set_random_seed(42)


def MobileNet(input_size=128, numclass=196):

    mobile_model = tf.keras.applications.MobileNet(
        weights="imagenet",
        alpha=0.25,
        input_tensor=k.layers.Input(shape=(input_size, input_size, 3),
                                    name='feature'),
        include_top=False)

    for layer in mobile_model.layers:
        layer.trainable = True

    x = mobile_model.output
    x = k.layers.GlobalAveragePooling2D()(x)
    output = k.layers.Dense(numclass,
                            name='predictions',
                            kernel_initializer=k.initializers.he_normal(42))(x)
    model = k.Model(inputs=mobile_model.input,
                    outputs=output, name='facekeypoint')

    return model
