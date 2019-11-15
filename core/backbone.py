import core.layers as layers
import tensorflow as tf


def DarkNet53(input_data, trainable):

    with tf.variable_scope('darknet'):

        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 3, 32), 
            trainable=trainable, name='conv0')
        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 32, 64),
            trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = layers.residual_block(input_data, input_channel=64, filter_num1=32, 
                filter_num2=64, trainable=trainable, name='residual%d' %(i+0))

        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 64, 128),
                trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = layers.residual_block(input_data, 128, 64, 128, 
                trainable=trainable, name='residual%d' %(i+1))

        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 128, 256),
                trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = layers.residual_block(input_data, 256, 128, 256, 
                trainable=trainable, name='residual%d' %(i+3))

        route_1 = input_data
        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 256, 512),
                trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = layers.residual_block(input_data, 512, 256, 512, 
                trainable=trainable, name='residual%d' %(i+11))

        route_2 = input_data
        input_data = layers.convolution_layer(input_data, filters_shape=(3, 3, 512, 1024),
                trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = layers.residual_block(input_data, 1024, 512, 1024, 
                trainable=trainable, name='residual%d' %(i+19))

        return route_1, route_2, input_data




