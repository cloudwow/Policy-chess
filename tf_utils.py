from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from bad_model_exception import BadModelException
from tensorflow.python.training import moving_averages


def _assign_moving_average(orig_val, new_val, momentum, name):
    with tf.name_scope(name):
        scaled_diff = (1 - momentum) * (new_val - orig_val)
        return tf.assign_add(orig_val, scaled_diff)


def batch_norm_layer(net, is_training, scope, decay=0.9):
    with tf.variable_scope(scope):

        return tf.contrib.layers.batch_norm(
            net,
            center=False,
            scale=True,
            is_training=is_training,
            decay=decay,
            zero_debias_moving_mean=True,
            scope="bn")


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.tf.contrib.slim.ops._two_element_tuple
  cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                        tf.minimum(shape[2], kernel_size[1])])
  """
    shape = input_tensor.get_shape().as_list()

    if len(shape) < 3 or shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [
            min(shape[1], kernel_size[0]),
            min(shape[2], kernel_size[1])
        ]
    return kernel_size_out


def avg_pool2d(net, shape, name, stride=1, padding="SAME"):

    return tf.nn.avg_pool(
        net,
        ksize=[1, stride, stride, 1],
        strides=[1, stride, stride, 1],
        padding=padding,
        name=name)


def conv2d(net, num_filters, filter_shape, name, stride=1, padding="SAME"):
    conv_filt_shape = [
        filter_shape[0],
        filter_shape[1],
        int(net.shape[3]),  #3 or 1?
        int(num_filters)
    ]

    # initialise weights and bias for the filter
    weights = tf.Variable(
        tf.truncated_normal(conv_filt_shape, stddev=0.01),
        name=name + "_weights")

    # setup the convolutional layer operation
    net = tf.nn.conv2d(net, weights, [1, stride, stride, 1], padding=padding)

    return net


def max_pool2d(net, shape, stride, padding, name):
    with tf.variable_scope("max_pool"):

        # now perform max pooling
        ksize = [1, shape[0], shape[1], 1]
        strides = [1, stride, stride, 1]
        net = tf.nn.max_pool(net, ksize=ksize, strides=strides, padding=padding)
    return net


def bias_layer(net):
    biases = tf.Variable(
        tf.constant(0.0, shape=[int(net.shape[-1])], dtype=tf.float32),
        trainable=True,
        name='biases')
    # bias = tf.Variable(tf.truncated_normal([int(net.shape[-1])], 0.1)),
    net = tf.nn.bias_add(net, biases)
    return net


def flat_size(net):
    flat_size = net.shape[1]
    for i in range(2, 999):
        if len(net.shape) <= i:
            break
        flat_size *= net.shape[i]
    return int(flat_size)


def dense_layer(net, size):
    if len(net.shape) > 2:
        raise BadModelException("Dense module require a flat input")
    flat_size = int(net.shape[1])

    bd1b = tf.Variable(
        tf.truncated_normal([size], stddev=0.01), name='dense_bias')
    wd1b = tf.Variable(
        tf.truncated_normal([flat_size, size], stddev=0.1), name='dense_weight')
    net = tf.matmul(net, wd1b) + bd1b

    return net


def flatten(net):
    flat_size = net.shape[1]
    for i in range(2, 999):
        if len(net.shape) <= i:
            break
        flat_size *= net.shape[i]
    flat_size = int(flat_size)
    net = tf.reshape(net, [-1, flat_size])

    return net
