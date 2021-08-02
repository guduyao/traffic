from tensorflow import keras
from tensorflow.python.keras.regularizers import l2
import tensorflow as tf

DECAY = 0.0005
INPUT_SHAPE = (416, 416, 3)
NUM_CLASSES = 80


def DarkNet_conv(x, filters, kernel_size, stride=1, bn=True):
    """
    :param x: input x
    :param filters: Conv2D filters
    :param kernel_size: Conv2D kernel_size
    :param stride: Conv2D stride
    :param bn: Use Conv2D BatchNormalization,default is True
    :return:
    """
    if stride == 1:
        padding = 'same'
    else:
        x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'

    x = keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        use_bias=not bn,
        kernel_regularizer=l2(DECAY),
        padding=padding,
    )(x)

    if bn:
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

    return x


def DarkNet_residual(x, filters):
    """
    :param x: input x
    :param filters: DarkNet_conv filters
    :return: residual x
    """
    orgi_x = x
    x = DarkNet_conv(x, filters // 2, kernel_size=1)
    x = DarkNet_conv(x, filters, kernel_size=3)
    x = keras.layers.Add()([orgi_x, x])
    return x


def DarkNet_block(x, filters, nums):
    """
    :param x: input x
    :param filters: Darknet_conv filters
    :param nums: Number of cycles
    :return: x
    """
    x = DarkNet_conv(x, filters=filters, kernel_size=3, stride=2)
    for _ in range(nums):
        x = DarkNet_residual(x, filters)
    return x


def DarkNet53(**kwargs):
    """
    :param kwargs: keras.Model(**kwargs)
    :return: darknet53 conv model->return feat1,feat2,feat3
    """
    keras.layers.Input(INPUT_SHAPE)
    x = inputs = keras.layers.Input(INPUT_SHAPE)
    x = DarkNet_conv(inputs, 32, kernel_size=3)
    x = DarkNet_block(x, 64, nums=1)
    x = DarkNet_block(x, 128, nums=2)
    x = feat1 = DarkNet_block(x, 256, nums=8)
    x = feat2 = DarkNet_block(x, 512, nums=8)
    x = feat3 = DarkNet_block(x, 1024, nums=4)
    return keras.Model(inputs, outputs=(feat1, feat2, feat3), **kwargs)


if __name__ == '__main__':
    darknet = DarkNet53()
    darknet.summary()
