"""Model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers
from .helpers import cast_and_norm, CustomConv3D

def build_model(input_shape, target_size, dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=3, padding='same', activation='relu')
    model = tf.keras.models.Sequential([
        layers.Lambda(cast_and_norm),
        layers.Conv3D(16, **conv_args, input_shape=input_shape),
        #layers.Lambda(lambda x: CustomConv3D(x, 16, **conv_args)),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(32, **conv_args),
        #layers.Lambda(lambda x: CustomConv3D(x, 32, **conv_args)),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(64, **conv_args),
        #layers.Lambda(lambda x: CustomConv3D(x, 64, **conv_args)),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(128, **conv_args),
        #layers.Lambda(lambda x: CustomConv3D(x, 128, **conv_args)),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(256, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout),

        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),

        layers.Dense(target_size)
    ])

    return model