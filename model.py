from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Reshape, Permute, Activation
from tensorflow.keras.layers import BatchNormalization
from keras import backend as K
def r2_unet(input_shape=(256, 256, 1), num_classes=7):
    inputs = Input(input_shape)

    # Encoding path
    conv1 = residual_conv_block(inputs, 16)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = residual_conv_block(pool1, 32)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = residual_conv_block(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = residual_conv_block(pool3, 128)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = residual_conv_block(pool4, 256)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = residual_conv_block(pool5, 512)

    # Decoding path
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv5], axis=-1)
    conv7 = residual_conv_block(up7, 256)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv4], axis=-1)
    conv8 = residual_conv_block(up8, 128)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv3], axis=-1)
    conv9 = residual_conv_block(up9, 64)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv2], axis=-1)
    conv10 = residual_conv_block(up10, 32)

    up11 = concatenate([UpSampling2D(size=(2, 2))(conv10), conv1], axis=-1)
    conv11 = residual_conv_block(up11, 16)

    conv12 = Conv2D(num_classes, (1, 1))(conv11)
    conv12 = Reshape((-1, num_classes))(conv12)
    conv12 = Permute((2, 1))(conv12)
    conv12 = Activation('softmax')(conv12)

    model = Model(inputs=inputs, outputs=conv12)
    return model
def residual_conv_block(inputs, num_filters):
    conv1 = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(num_filters, (3, 3), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    shortcut = Conv2D(num_filters, (1, 1), padding='same')(inputs)
    shortcut = BatchNormalization()(shortcut)

    output = concatenate([conv2, shortcut], axis=-1)

    return output
