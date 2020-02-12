import tensorflow as tf
import tensorflow.keras.layers as layers

def cast_and_norm(x):
     x=tf.cast(x, tf.float32)
     return x/tf.reduce_mean(x)

def conv3d_to_conv2d(x,d,c):
    return tf.reshape(x, [-1,d,d,c])

def conv2d_to_conv3d_padded3(x,d,c):
    x=tf.reshape(x, [-1,d,d,d,3,c])
    x1=x[:,:,:,:,0,:]
    x2=x[:,:,:,:,1,:]
    x3=x[:,:,:,:,2,:]
    x1=tf.pad(x1, [[0,0],[1,0],[0,0],[0,0],[0,0]], 'SYMMETRIC')
    x3=tf.pad(x3, [[0,0],[0,1],[0,0],[0,0],[0,0]], 'SYMMETRIC')
    x1=x1[:,:-1,:,:,:]
    x3=x3[:,1:,:,:,:]
    #  the roll kernel is very slow (and produces slightly different results anyway)
    #x1=tf.roll(x1,1,axis=1)
    #x3=tf.roll(x3,-1,axis=1)
    #x1=tf.concat([x1[:,0:1,:,:,:], x1[:,:-1,:,:,:]], axis=1)
    #x3=tf.concat([x3[:,1:,:,:,:], x3[:,-1:,:,:,:]], axis=1)
    return tf.reshape(x1+x2+x3,[-1,d,d,d,c])

def conv2d_to_conv3d_valid3(x,di,d,c):
    x=tf.reshape(x, [-1,di,d,d,3,c])
    x1=x[:,:-2,:,:,0,:]
    x2=x[:,1:-1,:,:,1,:]
    x3=x[:,2:,:,:,2,:]
    return tf.reshape(x1+x2+x3,[-1,d,d,d,c])

def conv2d_to_conv3d_valid2(x, di, d, c):
    x=tf.reshape(x, [-1,di,d,d,2,c])
    x1=x[:,:-1,:,:,0,:]
    x2=x[:,1:,:,:,1,:]
    return tf.reshape(x1+x2,[-1,d,d,d,c])

# MIOpen implements Conv3D via fallback to GEMM, which is not always optimal.
# This is an alternate implementation.
def CustomConv3D(x, out_features, kernel_size=3, padding='same', activation=None):
    dim=x.shape[-2]
    in_features=x.shape[-1]
    x = layers.Lambda(lambda x: conv3d_to_conv2d(x,dim,in_features))(x)
    x = layers.Conv2D(kernel_size*out_features, kernel_size=kernel_size, padding=padding)(x)
    dim_out=x.shape[-2]
    if kernel_size==2 and padding=='valid':
        x=layers.Lambda(lambda x: conv2d_to_conv3d_valid2(x,dim,dim_out,out_features))(x)
    elif kernel_size==3 and padding=='valid':
        x=layers.Lambda(lambda x: conv2d_to_conv3d_valid3(x,dim,dim_out,out_features))(x)
    elif kernel_size==3 and padding=='same':
        x=layers.Lambda(lambda x: conv2d_to_conv3d_padded3(x,dim,out_features))(x)
    else:
        print("Unsupported combination", kernel_size, padding)
        return None
    if activation==None:
        return x
    elif activation=='relu':
        return tf.nn.relu(x)
    else:
        print("Unsupported activation", activation)
        return None