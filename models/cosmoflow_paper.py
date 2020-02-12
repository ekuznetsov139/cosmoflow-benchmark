"""Model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers
from .helpers import cast_and_norm
def scale_1p2(x):
    return x*1.2

def build_model(input_shape, target_size, dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=2, padding='valid')

    model = tf.keras.models.Sequential([
        layers.Lambda(cast_and_norm),
        layers.Conv3D(16, input_shape=input_shape, kernel_size=3, padding="valid"),#128 -> 126
        layers.LeakyReLU(),
        layers.AvgPool3D(pool_size=2),# 126 -> 63

        layers.Conv3D(32, kernel_size=4, padding="valid"), # 63 -> 60
        layers.LeakyReLU(),
        layers.AvgPool3D(pool_size=2), # 60 -> 30

        layers.Conv3D(64, kernel_size=4, padding="valid"), # 30 -> 27
        layers.LeakyReLU(),
        layers.AvgPool3D(pool_size=2), # 27 -> 13

        layers.Conv3D(128, kernel_size=3, strides=2, padding="valid"),# 13 -> 6
        layers.LeakyReLU(),

        layers.Conv3D(256, kernel_size=3, padding="valid"), # 6 -> 4
        layers.LeakyReLU(),

        layers.Conv3D(256, kernel_size=2, padding="valid"), # 4 -> 3
        layers.LeakyReLU(),

        layers.Conv3D(256, kernel_size=2, padding="valid"), # 3 -> 2
        layers.LeakyReLU(),

        layers.Flatten(),
        layers.Dropout(dropout),

        layers.Dense(2048),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(256),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(target_size),
#        layers.Dense(target_size, activation='tanh'),
#        layers.Lambda(scale_1p2)
    ])

    return model
