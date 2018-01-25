"""Training script for the network."""

from __future__ import print_function

import os
import fnmatch
import random
import numpy as np
import tensorflow as tf
import constants
import tf_utils
labels = []


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.01)
        result = tf.Variable(initial)
        variable_summaries(result)
        return result


def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.01, shape=shape)
        result = tf.Variable(initial)
        variable_summaries(result)
        return result


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def model(batch_size, is_training):
    # network weights
    data_placeholder = tf.placeholder(
        tf.float32,
        shape=(batch_size, constants.IMAGE_SIZE, constants.IMAGE_SIZE,
               constants.FEATURE_PLANES))

    net = data_placeholder

    # hidden layers
    for i in range(0, 4):
        with tf.name_scope("module_" + str(i)):
            with tf.name_scope("branch_1_" + str(i)):
                branch_1 = net
                for j in range(0, 3):

                    branch_1 = tf_utils.conv2d(branch_1, 64, [3, 3], "branch_1")
            with tf.name_scope("branch_2_" + str(i)):

                branch_2 = net
                for j in range(0, 3):
                    with tf.name_scope(
                            "barnch_2_conv_" + str(j) + "_" + str(i)):
                        branch_2 = tf_utils.conv2d(branch_2, 192, [8, 8],
                                                   "conv")

                        branch_2 = tf_utils.batch_norm_layer(
                            branch_2,
                            is_training,
                            decay=0.9,
                            scope="branch_2_batchnorm_" + str(j) + "_" + str(i))
                        branch_2 = tf.nn.relu(branch_2)
            net = tf.concat(axis=3, values=[branch_1, branch_2])

    net = tf_utils.flatten(net)
    with tf.name_scope("fc_1"):
        net = tf.nn.relu(tf_utils.dense_layer(net, constants.HIDDEN))


#    with tf.name_scope("fc_1"):
#        h_flat = tf.reshape(h_conv3, [-1, constants.HIDDEN])
#        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# readout layer
    logits = tf_utils.dense_layer(net, constants.LABEL_SIZE)

    return data_placeholder, logits
