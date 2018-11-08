# @author : Abhishek R S

import os
import h5py
import numpy as np
import tensorflow as tf

'''
SegNet
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition]
  (https://arxiv.org/abs/1409.1556)
- [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
- [Bayesian SegNet](https://arxiv.org/pdf/1511.02680.pdf)
- [SegNet Project](http://mi.eng.cam.ac.uk/projects/segnet/)

# Pretrained model weights
- [Download pretrained vgg-16 model]
  (https://github.com/fchollet/deep-learning-models/releases/)
'''


class SegNet:
    def __init__(self, pretrained_weights, is_training, data_format='channels_first', num_classes=15):
        self._weights_h5 = h5py.File(pretrained_weights, 'r')
        self._is_training = is_training
        self._data_format = data_format
        self._num_classes = num_classes
        self._padding = 'SAME'
        self._encoder_conv_strides = [1, 1, 1, 1]
        self._feature_map_axis = None
        self._encoder_data_format = None
        self._encoder_pool_kernel = None
        self._encoder_pool_strides = None
        self._initializer = tf.contrib.layers.xavier_initializer_conv2d()

        '''
        based on the data format set appropriate pool_kernel and pool_strides
        always use channels_first i.e. NCHW as the data format on a GPU
        '''

        if data_format == 'channels_first':
            self._encoder_data_format = 'NCHW'
            self._encoder_pool_kernel = [1, 1, 2, 2]
            self._encoder_pool_strides = [1, 1, 2, 2]
            self._feature_map_axis = 1
        else:
            self._encoder_data_format = 'NHWC'
            self._encoder_pool_kernel = [1, 2, 2, 1]
            self._encoder_pool_strides = [1, 2, 2, 1]
            self._feature_map_axis = -1

    # build vgg-16 encoder
    def vgg16_encoder(self, features):

        # input : BGR format with image_net mean subtracted
        # bgr mean : [103.939, 116.779, 123.68]

        if self._data_format == 'channels_last':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        # Stage 1
        self.conv1_1 = self._conv_block(features, 'block1_conv1')
        self.conv1_2 = self._conv_block(self.conv1_1, 'block1_conv2')
        self.pool1 = self._maxpool_layer(self.conv1_2, name='pool1')

        # Stage 2
        self.conv2_1 = self._conv_block(self.pool1, 'block2_conv1')
        self.conv2_2 = self._conv_block(self.conv2_1, 'block2_conv2')
        self.pool2 = self._maxpool_layer(self.conv2_2, name='pool2')

        # Stage 3
        self.conv3_1 = self._conv_block(self.pool2, 'block3_conv1')
        self.conv3_2 = self._conv_block(self.conv3_1, 'block3_conv2')
        self.conv3_3 = self._conv_block(self.conv3_2, 'block3_conv3')
        self.pool3 = self._maxpool_layer(self.conv3_3, name='pool3')

        # Stage 4
        self.conv4_1 = self._conv_block(self.pool3, 'block4_conv1')
        self.conv4_2 = self._conv_block(self.conv4_1, 'block4_conv2')
        self.conv4_3 = self._conv_block(self.conv4_2, 'block4_conv3')
        self.pool4 = self._maxpool_layer(self.conv4_3, name='pool4')

        # Stage 5
        self.conv5_1 = self._conv_block(self.pool4, 'block5_conv1')
        self.conv5_2 = self._conv_block(self.conv5_1, 'block5_conv2')
        self.conv5_3 = self._conv_block(self.conv5_2, 'block5_conv3')
        self.pool5 = self._maxpool_layer(self.conv5_3, name='pool5')

    # define the bayesian decoder with transposed convolution upsampling
    def segnet_bayesian(self):
        self.decoder1 = self._get_decoder_block_tr_conv(
            self.pool5, 512, 3, name='decoder1_')
        self.dropout1 = self._get_dropout_layer(
            self.decoder1, name='decoder1_dropout')
        self.decoder2 = self._get_decoder_block_tr_conv(
            self.dropout1, 512, 3, name='decoder2_')
        self.dropout2 = self._get_dropout_layer(
            self.decoder2, name='decoder2_dropout')
        self.decoder3 = self._get_decoder_block_tr_conv(
            self.dropout2, 256, 3, name='decoder3_')
        self.dropout3 = self._get_dropout_layer(
            self.decoder3, name='decoder3_dropout')
        self.decoder4 = self._get_decoder_block_tr_conv(
            self.dropout3, 128, 2, name='decoder4_')
        self.dropout4 = self._get_dropout_layer(
            self.decoder4, name='decoder4_dropout')
        self.decoder5 = self._get_decoder_block_tr_conv(
            self.dropout4, 64, 2, name='decoder5_')

        self.logits = self._get_conv2d_layer(self.decoder5, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # define the decoder with transposed convolution upsampling
    def segnet_tr_conv(self):
        self.decoder1 = self._get_decoder_block_tr_conv(
            self.pool5, 512, 3, name='decoder1_')
        self.decoder2 = self._get_decoder_block_tr_conv(
            self.decoder1, 512, 3, name='decoder2_')
        self.decoder3 = self._get_decoder_block_tr_conv(
            self.decoder2, 256, 3, name='decoder3_')
        self.decoder4 = self._get_decoder_block_tr_conv(
            self.decoder3, 128, 2, name='decoder4_')
        self.decoder5 = self._get_decoder_block_tr_conv(
            self.decoder4, 64, 2, name='decoder5_')

        self.logits = self._get_conv2d_layer(self.decoder5, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # define the decoder with bilinear upsampling
    def segnet_bilinear(self):
        self.decoder1 = self._get_decoder_block_bilinear(
            self.pool5, 512, 3, name='decoder1_')
        self.decoder2 = self._get_decoder_block_bilinear(
            self.decoder1, 512, 3, name='decoder2_')
        self.decoder3 = self._get_decoder_block_bilinear(
            self.decoder2, 256, 3, name='decoder3_')
        self.decoder4 = self._get_decoder_block_bilinear(
            self.decoder3, 128, 2, name='decoder4_')
        self.decoder5 = self._get_decoder_block_bilinear(
            self.decoder4, 64, 2, name='decoder5_')

        self.logits = self._get_conv2d_layer(self.decoder5, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # define the decoder with nearest neighbor upsampling
    def segnet_nn(self):
        self.decoder1 = self._get_decoder_block_nn(
            self.pool5, 512, 3, name='decoder1_')
        self.decoder2 = self._get_decoder_block_nn(
            self.decoder1, 512, 3, name='decoder2_')
        self.decoder3 = self._get_decoder_block_nn(
            self.decoder2, 256, 3, name='decoder3_')
        self.decoder4 = self._get_decoder_block_nn(
            self.decoder3, 128, 2, name='decoder4_')
        self.decoder5 = self._get_decoder_block_nn(
            self.decoder4, 64, 2, name='decoder5_')

        self.logits = self._get_conv2d_layer(self.decoder5, self._num_classes, [
                                             1, 1], [1, 1], name='logits')

    # return decoder block for transposed convolution upsampling
    def _get_decoder_block_tr_conv(self, features, num_out_features, num_intermediate_layers, name='decoder_'):
        _up = self._get_conv2d_transpose_layer(features, num_out_features, [
                                               2, 2], [2, 2], name=name + 'tr_conv')
        inputs = _up

        for i in range(num_intermediate_layers):
            _conv = self._get_conv2d_layer(inputs, num_out_features, [3, 3], [
                                           1, 1], name=name + 'conv' + str(i + 1))
            _bn = self._get_batchnorm_layer(
                _conv, name=name + 'bn' + str(i + 1))
            _relu = self._get_relu_activation(
                _bn, name=name + 'relu' + str(i + 1))
            inputs = _relu

        return _relu

    # return decoder block for bilinear upsampling
    def _get_decoder_block_bilinear(self, features, num_out_features, num_intermediate_layers, name='decoder_'):
        if self._data_format == 'channels_first':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        up_size = 2 * tf.shape(features)[1:3]
        _up = tf.image.resize_bilinear(
            features, size=up_size, name=name + 'up_bilinear')

        if self._data_format == 'channels_first':
            _up = tf.transpose(_up, perm=[0, 3, 1, 2])

        inputs = _up

        for i in range(num_intermediate_layers):
            _conv = self._get_conv2d_layer(inputs, num_out_features, [3, 3], [
                                           1, 1], name=name + 'conv' + str(i + 1))
            _bn = self._get_batchnorm_layer(
                _conv, name=name + 'bn' + str(i + 1))
            _relu = self._get_relu_activation(
                _bn, name=name + 'relu' + str(i + 1))
            inputs = _relu

        return _relu

    # return decoder block for nearest neighbor upsampling
    def _get_decoder_block_nn(self, features, num_out_features, num_intermediate_layers, name='decoder_'):
        if self._data_format == 'channels_first':
            features = tf.transpose(features, perm=[0, 2, 3, 1])

        up_size = 2 * tf.shape(features)[1:3]
        _up = tf.image.resize_nearest_neighbor(
            features, size=up_size, name=name + 'up_nearest_neighbour')

        if self._data_format == 'channels_first':
            _up = tf.transpose(_up, perm=[0, 3, 1, 2])

        inputs = _up

        for i in range(num_intermediate_layers):
            _conv = self._get_conv2d_layer(inputs, num_out_features, [3, 3], [
                                           1, 1], name=name + 'conv' + str(i + 1))
            _bn = self._get_batchnorm_layer(
                _conv, name=name + 'bn' + str(i + 1))
            _relu = self._get_relu_activation(
                _bn, name=name + 'relu' + str(i + 1))
            inputs = _relu

        return _relu

    # return convolution2d layer
    def _get_conv2d_layer(self, inputs, num_filters, kernel_size, strides, name='conv'):
        return tf.layers.conv2d(inputs=inputs, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return convolution2d_transpose layer
    def _get_conv2d_transpose_layer(self, inputs, num_filters, kernel_size, strides, name='conv_tr'):
        return tf.layers.conv2d_transpose(inputs=inputs, filters=num_filters, kernel_size=kernel_size, strides=strides, padding=self._padding, data_format=self._data_format, kernel_initializer=self._initializer, name=name)

    # return relu activation function
    def _get_relu_activation(self, inputs, name='relu'):
        return tf.nn.relu(inputs, name=name)

    # return dropout layer
    def _get_dropout_layer(self, inputs, rate=0.5, name='dropout'):
        return tf.layers.dropout(inputs=inputs, rate=rate, training=self._is_training, name=name)

    # return batch normalization layer
    def _get_batchnorm_layer(self, inputs, name='bn'):
        return tf.layers.batch_normalization(inputs, axis=self._feature_map_axis, training=self._is_training, name=name)

    #-------------------------------------#
    # pretrained vgg-16 encoder functions #
    #-------------------------------------#
    #-----------------------#
    # convolution2d layer   #
    #-----------------------#
    def _conv_block(self, input_layer, name):
        W = tf.constant(self._weights_h5[name][name + '_W_1:0'])
        b = self._weights_h5[name][name + '_b_1:0']
        b = tf.constant(np.reshape(b, (b.shape[0])))

        x = tf.nn.conv2d(input_layer, filter=W, strides=self._encoder_conv_strides,
                         padding=self._padding, data_format=self._encoder_data_format, name=name + '_conv')
        x = tf.nn.bias_add(
            x, b, data_format=self._encoder_data_format, name=name + '_bias')
        x = tf.nn.relu(x, name=name + '_relu')

        return x

    #-----------------------#
    # maxpool2d layer       #
    #-----------------------#
    def _maxpool_layer(self, input_layer, name):
        pool = tf.nn.max_pool(input_layer, ksize=self._encoder_pool_kernel, strides=self._encoder_pool_strides,
                              padding=self._padding, data_format=self._encoder_data_format, name=name)

        return pool
