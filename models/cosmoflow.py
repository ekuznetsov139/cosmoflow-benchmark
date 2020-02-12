"""Model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers
from .helpers import cast_and_norm, CustomConv3D
def scale_1p2(x):
    return x*1.2
#ef CustomConv3D(x, out_features, kernel_size, padding):
 

def build_model(input_shape, target_size, dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=2, padding='valid')
    pool_args = dict(pool_size=2)
    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Lambda(cast_and_norm),
        #layers.Conv3D(16, **conv_args),
        layers.Lambda(lambda x: CustomConv3D(x, 16, **conv_args)),
        layers.LeakyReLU(),
        layers.MaxPool3D(**pool_args),

        #layers.Conv3D(16, **conv_args),
        layers.Lambda(lambda x: CustomConv3D(x, 16, **conv_args)),
        layers.LeakyReLU(),
        layers.MaxPool3D(**pool_args),

        #layers.Conv3D(16, **conv_args),
        layers.Lambda(lambda x: CustomConv3D(x, 16, **conv_args)),
        layers.LeakyReLU(),
        layers.MaxPool3D(**pool_args),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(**pool_args),

        layers.Conv3D(16, **conv_args),
        layers.LeakyReLU(),
        layers.MaxPool3D(**pool_args),

        layers.Flatten(),
        layers.Dropout(dropout),

        layers.Dense(128),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(64),
        layers.LeakyReLU(),
        layers.Dropout(dropout),

        layers.Dense(target_size, activation='tanh'),
        layers.Lambda(scale_1p2)
    ])

    return model