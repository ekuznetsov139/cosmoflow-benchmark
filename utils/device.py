"""
Hardware/device configuration
"""

# System
import os

# Externals
import tensorflow as tf

def configure_session(intra_threads=32, inter_threads=2,
                      blocktime=1, affinity='granularity=fine,compact,1,0',
                      gpu=None):
    """Sets the thread knobs in the TF backend"""
    os.environ['KMP_BLOCKTIME'] = str(blocktime)
    os.environ['KMP_AFFINITY'] = affinity
    os.environ['OMP_NUM_THREADS'] = str(intra_threads)
    config = tf.compat.v1.ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads
    )
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
        tf.config.experimental.set_visible_devices(tf.config.experimental.list_physical_devices('GPU')[gpu],'GPU')
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
