"""Training script for the network."""

from __future__ import print_function

import os
import fnmatch
import random
import numpy as np
import tensorflow as tf

TRAIN_DIRECTORY = './data_train'
VALIDATION_DIRECTORY = './data_validation'
LABELS_DIRECTORY = './labels'
BATCH_SIZE = 50
IMAGE_SIZE = 8
FEATURE_PLANES = 8
LABEL_SIZE = 6100
FILTERS = 128
HIDDEN = 512
NUM_STEPS = 150001

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


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def model(batch_size):
    # network weights
    data_placeholder = tf.placeholder(tf.float32,
                                      shape=(batch_size,
                                             IMAGE_SIZE,
                                             IMAGE_SIZE,
                                             FEATURE_PLANES))

    W_conv1 = weight_variable([IMAGE_SIZE, IMAGE_SIZE, FEATURE_PLANES, FILTERS])
    b_conv1 = bias_variable([FILTERS])

    W_conv2 = weight_variable([5, 5, FILTERS, FILTERS])
    b_conv2 = bias_variable([FILTERS])

    W_conv3 = weight_variable([3, 3, FILTERS, FILTERS])
    b_conv3 = bias_variable([FILTERS])

    W_fc1 = weight_variable([HIDDEN, HIDDEN])
    b_fc1 = bias_variable([HIDDEN])

    W_fc2 = weight_variable([HIDDEN, LABEL_SIZE])
    b_fc2 = bias_variable([LABEL_SIZE])

    # hidden layers
    with tf.name_scope("conv_1"):
        h_conv1 = tf.nn.relu(conv2d(data_placeholder, W_conv1, 1) + b_conv1)
    with tf.name_scope("conv_2"):
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 3) + b_conv2)
    with tf.name_scope("conv_3"):
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    with tf.name_scope("pool"):
        h_pool3 = max_pool_2x2(h_conv3)
        h_flat = tf.reshape(h_pool3, [-1, HIDDEN])
    with tf.name_scope("fc_1"):
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    # readout layer
        logits = tf.matmul(h_fc1, W_fc2) + b_fc2

    return data_placeholder, logits

