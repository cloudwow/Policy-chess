"""Training script for the network."""

from __future__ import print_function

import os
import fnmatch
import random
import numpy as np
import tensorflow as tf
from termcolor import colored

import model
import constants
import encoder
import examples

labels = []
tf_policy_labels = tf.placeholder(
    tf.float32, shape=(constants.BATCH_SIZE, constants.LABEL_SIZE))
tf_value_labels = tf.placeholder(tf.float32, shape=(constants.BATCH_SIZE, 1))

# Training computation.
tf_train_dataset, policy_logits, value = model.model(constants.BATCH_SIZE, True,
                                                     2)
with tf.name_scope('cross_entropy'):

    with tf.name_scope('policy'):
        policy_diff = tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=tf_policy_labels)
        with tf.name_scope('policy_total'):
            policy_loss = tf.reduce_mean(policy_diff)
    # with tf.name_scope('value'):
    #     value_loss = tf.losses.absolute_difference(value, tf_value_labels, 20.0)

    # with tf.name_scope('loss_sum'):
    #     total_loss = tf.add(policy_loss, value_loss)
global_step_tensor = tf.Variable(0, name='global_step', trainable=False)

#tf.summary.scalar('total_loss', loss)
tf.summary.scalar('policy_loss', policy_loss)
#tf.summary.scalar('value_loss', value_loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=0.0001, epsilon=0.1).minimize(
            policy_loss, global_step=global_step_tensor)

# optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Predictions for the training, validation, and test data.

train_prediction = tf.nn.softmax(policy_logits)
labels_argmax = tf.argmax(tf_policy_labels, 1)
soft_argmax = tf.argmax(train_prediction, 1)
correct_prediction = tf.equal(labels_argmax, soft_argmax)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(constants.LOGDIR + '/train', sess.graph)
test_writer = tf.summary.FileWriter(constants.LOGDIR + '/test')

sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state(constants.CHECKPOINT_DIRECTORY)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])


def main():
    print('Training...')
    best_validation_loss = 999.0
    step = -1
    validation_batch_iterator = examples.generate_batches(
        constants.VALIDATION_DIRECTORY)
    for train_dataset in examples.generate_batches(constants.TRAIN_DIRECTORY):
        step += 1
        batch_data = []
        policy_labels = []
        value_labels = []
        for plane, policy_label, value_label in train_dataset:
            batch_data.append(plane)
            policy_labels.append(policy_label)
#            value_labels.append([value_label])
        if len(batch_data) != constants.BATCH_SIZE:
            print("bad sizes train dataset")
            print(len(batch_data))
            continue

        feed_dict = {
            tf_train_dataset: batch_data,
            tf_policy_labels: policy_labels,
            #           tf_value_labels: value_labels
        }
        summary, _, l, predictions = sess.run(
            [merged, optimizer, policy_loss, train_prediction],
            feed_dict=feed_dict)
        global_step = tf.train.global_step(sess, global_step_tensor)
        train_writer.add_summary(summary, global_step)
        #        print("global step: %d" % global_step)
        if (step % 50 == 0 and step > 0):

            print(colored('Minibatch loss at step %d: %f' % (step, l), "cyan"))
            print('Minibatch accuracy: %.1f%%' % accuracy(
                predictions, policy_labels))

            # We check accuracy with the validation data set
            validation_dataset = validation_batch_iterator.next()
            batch_valid_data = []
            batch_valid_policy_labels = []
            batch_valid_value_labels = []
            for plane, policy_label, value_label in validation_dataset:
                batch_valid_data.append(plane)
                batch_valid_policy_labels.append(policy_label)


#               batch_valid_value_labels.append([value_label])
            if len(batch_valid_data) != constants.BATCH_SIZE:
                print("bad sizes validation dataset")
                print(len(batch_valid_data))
                continue
            feed_dict_valid = {
                tf_train_dataset: batch_valid_data,
                tf_policy_labels: batch_valid_policy_labels,
                #                tf_value_labels: batch_valid_value_labels
            }

            summary, this_validation_loss, predictions_valid = sess.run(
                [merged, policy_loss, train_prediction],
                feed_dict=feed_dict_valid)

            test_writer.add_summary(summary, global_step)
            print('validation loss at step %d: %f' % (step,
                                                      this_validation_loss))
            print('Validation accuracy: %.1f%%' % accuracy(
                predictions_valid, batch_valid_policy_labels))

            if this_validation_loss < best_validation_loss:
                print(colored("saving model", "green"))
                saver.save(
                    sess,
                    constants.CHECKPOINT_DIRECTORY + '/chess-dqn',
                    global_step=global_step)
                best_validation_loss = this_validation_loss

if __name__ == '__main__':
    main()
