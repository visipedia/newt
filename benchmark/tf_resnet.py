"""ResNet50 model for Keras.
Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.
Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

layers = tf.keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return regularizers.l2(l2_weight_decay) if use_l2_regularizer else None

def fixed_padding(inputs, kernel_size, data_format='channels_last'):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: `Tensor` of size `[batch, channels, height, width]` or
            `[batch, height, width, channels]` depending on `data_format`.
        kernel_size: `int` kernel size to be used for `conv2d` or max_pool2d`
            operations. Should be a positive integer.
        data_format: `str` either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.
    Returns:
        A padded `Tensor` of the same `data_format` with size either intact
        (if `kernel_size == 1`) or padded (if `kernel_size > 1`).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])

    return padded_inputs


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5,
                   batch_norm_trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Kernel: 1
    # Strides: 1

    x = layers.Conv2D(
        filters1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2a')(
            x)
    x = layers.Activation('relu')(x)

    # Kernel: 3
    # Strides: 1

    x = layers.Conv2D(
        filters2,
        kernel_size=kernel_size,
        strides=(1,1),
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # Kernel: 1
    # Strides: 1

    x = layers.Conv2D(
        filters3,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5,
               batch_norm_trainable=True):
    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    # Kernel: 1
    # Stride: Dynamic

    shortcut = layers.Conv2D(
        filters3,
        kernel_size=(1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '1')(shortcut)


    # Kernel: 1
    # Stride: 1

    x = layers.Conv2D(
        filters1,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    # Kernel: 3
    # Strides: Dynamic

    if strides[0] > 1:
        x = fixed_padding(x, kernel_size, data_format=backend.image_data_format())
        padding = 'valid'
    else:
        padding = 'same'

    x = layers.Conv2D(
        filters2,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    # Kernel: 1
    # Stride: 1

    x = layers.Conv2D(
        filters3,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name=bn_name_base + '2c')(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x




def resnet50(
            include_top=True,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            batch_size=None,
            use_l2_regularizer=True,
            rescale_inputs=False,
            batch_norm_decay=0.9,
            batch_norm_epsilon=1e-5,
            batch_norm_trainable=True,
            width_multiplier=1):
    """Instantiates the ResNet50 architecture.
    Args:
        num_classes: `int` number of classes for image classification.
        batch_size: Size of the batches for each step.
        use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
        rescale_inputs: whether to rescale inputs from 0 to 1.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        A Keras model instance.
    """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        batch_norm_trainable=batch_norm_trainable)

    #x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = fixed_padding(x, 7, backend.image_data_format())
    x = layers.Conv2D(
        64 * width_multiplier,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='conv1')(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        trainable=batch_norm_trainable,
        name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, np.array([64, 64, 256]) * width_multiplier, stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, np.array([64, 64, 256]) * width_multiplier, stage=2, block='b', **block_config)
    x = identity_block(x, 3, np.array([64, 64, 256]) * width_multiplier, stage=2, block='c', **block_config)

    x = conv_block(x, 3, np.array([128, 128, 512]) * width_multiplier, stage=3, block='a', **block_config)
    x = identity_block(x, 3, np.array([128, 128, 512]) * width_multiplier, stage=3, block='b', **block_config)
    x = identity_block(x, 3, np.array([128, 128, 512]) * width_multiplier, stage=3, block='c', **block_config)
    x = identity_block(x, 3, np.array([128, 128, 512]) * width_multiplier, stage=3, block='d', **block_config)

    x = conv_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='a', **block_config)
    x = identity_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='b', **block_config)
    x = identity_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='c', **block_config)
    x = identity_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='d', **block_config)
    x = identity_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='e', **block_config)
    x = identity_block(x, 3, np.array([256, 256, 1024]) * width_multiplier, stage=4, block='f', **block_config)

    x = conv_block(x, 3, np.array([512, 512, 2048]) * width_multiplier, stage=5, block='a', **block_config)
    x = identity_block(x, 3, np.array([512, 512, 2048]) * width_multiplier, stage=5, block='b', **block_config)
    x = identity_block(x, 3, np.array([512, 512, 2048]) * width_multiplier, stage=5, block='c', **block_config)

    x = layers.GlobalAveragePooling2D()(x)

    if include_top:
        x = layers.Dense(
            classes,
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name='fc1000')(
                x)

        # A softmax that is followed by the model loss must be done cannot be done
        # in float16 due to numeric issues. So we pass dtype=float32.
        x = layers.Activation('softmax', dtype='float32')(x)


    # Create model.
    model = models.Model(img_input, x, name='resnet50')

    if weights is not None:
        model.load_weights(weights)

    return model