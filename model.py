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
    with tf.name_scope("first_layer"):

        net = tf_utils.conv2d(net, 256, [3, 3], "first_layer")
        net = tf_utils.batch_norm_layer(
            net, is_training, decay=0.95, scope="first_layer_batch_n0rm_")
        net = tf.nn.relu(net)

    for i in range(0, 8):
        with tf.name_scope("module_" + str(i)):
            branches = []

            with tf.name_scope("branch_skip_" + str(i)):
                skip_branch = tf_utils.conv2d(net, 64, [1, 1], "skip_conv_1x1")
                branches.append(skip_branch)

            with tf.name_scope("branch_0_" + str(i)):
                net = tf_utils.conv2d(net, 256, [3, 3], "first_layer")
                net = tf_utils.batch_norm_layer(
                    net,
                    is_training,
                    decay=0.95,
                    scope="first_layer_batch_n0rm_" + str(i))
                net = tf.nn.relu(net)
                net = tf_utils.conv2d(net, 256, [3, 3], "second_layer")
                net = tf_utils.batch_norm_layer(
                    net,
                    is_training,
                    decay=0.95,
                    scope="scond_layer_batch_n0rm_" + str(i))
                branches.append(net)
            net = tf.concat(axis=3, values=branches)
            net = tf.nn.relu(net)

    return data_placeholder, policy_head(net, is_training), value_head(
        net, is_training)


def value_head(net, is_training):
    with tf.name_scope("value_head"):

        net = tf_utils.conv2d(net, 128, [1, 1], "1x1_reduce")
        net = tf_utils.batch_norm_layer(
            net, is_training, decay=0.95, scope="value_bn")
        net = tf.nn.relu(net)

        net = tf_utils.flatten(net)
        with tf.name_scope("fc_1"):
            net = tf.nn.relu(tf_utils.dense_layer(net, 20))

        net = tf_utils.dense_layer(net, 1)
        return net


def policy_head(net, is_training):
    with tf.name_scope("policy_head"):

        net = tf_utils.conv2d(net, 128, [1, 1], "1x1_reduce")
        net = tf_utils.batch_norm_layer(
            net, is_training, decay=0.95, scope="policy_bn")
        net = tf.nn.relu(net)

        net = tf_utils.flatten(net)
        logits = tf_utils.dense_layer(net, constants.LABEL_SIZE)
        return logits
