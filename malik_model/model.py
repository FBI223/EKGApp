import tensorflow as tf
from tensorflow.keras import layers, models

class SelfONN1D(layers.Layer):
    def __init__(self, filters, kernel_size, Q=3, strides=1, padding='same'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.Q = Q
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.W = self.add_weight(shape=(self.Q, self.kernel_size, in_channels, self.filters),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, x):
        outputs = 0
        for q in range(1, self.Q + 1):
            x_q = tf.math.pow(x, q)
            conv = tf.nn.conv1d(x_q, self.W[q - 1], stride=self.strides, padding=self.padding.upper())
            outputs += conv
        return outputs

def build_selfonn_model(input_shape=(128, 2), num_classes=5, Q=7):
    inputs = tf.keras.Input(shape=input_shape)

    x = SelfONN1D(filters=16, kernel_size=15, Q=Q)(inputs)
    x = layers.AveragePooling1D(pool_size=6)(x)
    x = tf.nn.tanh(x)

    x = SelfONN1D(filters=8, kernel_size=15, Q=Q)(x)
    x = layers.AveragePooling1D(pool_size=5)(x)
    x = tf.nn.tanh(x)

    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='tanh')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

