"""Training script for the network."""

from __future__ import print_function

import os
import fnmatch
import random
import numpy as np
import tensorflow as tf
import model
import constants

labels = []
tf_train_labels = tf.placeholder(tf.float32,
                                 shape=(constants.BATCH_SIZE,
                                 constants.LABEL_SIZE))

# Training computation.
tf_train_dataset, logits = model.model(constants.BATCH_SIZE)
with tf.name_scope('cross_entropy'):
  diff = tf.nn.softmax_cross_entropy_with_logits(
      logits=logits,
      labels=tf_train_labels)
  with tf.name_scope('total'):
    loss = tf.reduce_mean(diff)

global_step_tensor = tf.Variable(0, name='global_step',trainable=False)

tf.summary.scalar('cross_entropy', loss)
global_step_tensor = tf.Variable(0,name='global_step',trainable=False)
     
# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step_tensor)
# optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(constants.LOGDIR + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(constants.LOGDIR + '/test')

sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state(constants.CHECKPOINT_DIRECTORY )
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print ("Successfully loaded:", checkpoint.model_checkpoint_path)


def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
            return files


def _read_text(filename, batch_size):
    with open(filename) as f_in:
        lines = (line.rstrip() for line in f_in)
        # drop empty lines
        lines = list(line for line in lines if line)
        return random.sample(lines, batch_size)


def generate_batch(batch_size, directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    random.shuffle(files)
    for filename in files:
        text = _read_text(filename, batch_size)
        yield text


def read_labels(directory, pattern):
    '''Generator that yields text raw from the directory.'''
    files = find_files(directory, pattern)
    labels_array = []
    for filename in files:
        with open(filename) as f:
            lines = str(f.readlines()[0]).split(" ")
            for label in lines:
                if(label != " " and label != '\n'):
                    labels_array.append(label)
    return labels_array


def reformat(datas, labels):
    games = list(datas)[0]
    for game in games:
        try:
            board_state = (game.split(":")[0]).replace("/", "")
            label_state = (game.split(":")[1]).replace("\n", "")
        except:
            print("error parsing game: '"+game+"'")
            continue
        label = np.zeros(constants.LABEL_SIZE)
        for i in range(constants.LABEL_SIZE):
            if(label_state == labels[i]):
                label[i] = 1.

        # All pieces plane
        board_pieces = list(board_state.split(" ")[0])
        board_pieces = [ord(val) for val in board_pieces]
        board_pieces = np.reshape(board_pieces, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only spaces plane
        board_blank = [int(val == '1') for val in board_state.split(" ")[0]]
        board_blank = np.reshape(board_blank, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only white plane
        board_white = [int(val.isupper()) for val in board_state.split(" ")[0]]
        board_white = np.reshape(board_white, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only black plane
        board_black = [int(not val.isupper() and val != '1') for val in board_state.split(" ")[0]]
        board_black = np.reshape(board_black, (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # One-hot integer plane current player turn
        current_player = board_state.split(" ")[1]
        current_player = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), int(current_player == 'w'), dtype=int)
        # One-hot integer plane extra data
        extra = board_state.split(" ")[4]
        extra = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), int(extra), dtype=int)
        # One-hot integer plane move number
        move_number = board_state.split(" ")[5]
        move_number = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), int(move_number), dtype=int)
        # Zeros plane
        zeros = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), 0, dtype=int)

        planes = np.vstack((np.copy(board_pieces),
                            np.copy(board_white),
                            np.copy(board_black),
                            np.copy(board_blank),
                            np.copy(current_player),
                            np.copy(extra),
                            np.copy(move_number),
                            np.copy(zeros)))
        planes = np.reshape(planes, (constants.IMAGE_SIZE, constants.IMAGE_SIZE, constants.FEATURE_PLANES))
        yield (planes, label)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


def main():
    labels = read_labels(constants.LABELS_DIRECTORY, "*.txt")
    print('Training...')
    for step in range(constants.NUM_STEPS):
        train_batch = generate_batch(constants.BATCH_SIZE, constants.TRAIN_DIRECTORY, "*.txt")
        train_dataset = reformat(train_batch, labels)
        batch_data = []
        batch_labels = []
        for plane, label in train_dataset:
            batch_data.append(plane)
            batch_labels.append(label)
        if len(batch_data) != constants.BATCH_SIZE:
            print("bad sizes train dataset")
            print(len(batch_data))
            continue

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        summary, _, l, predictions = sess.run(
          [merged, optimizer, loss, train_prediction], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        global_step = tf.train.global_step(sess, global_step_tensor)
        print("global step: %d" % global_step)
        if (step % 100 == 0 and step > 0):

            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            
            # We check accuracy with the validation data set
            validation_batch = generate_batch(constants.BATCH_SIZE, constants.VALIDATION_DIRECTORY, "*.txt")
            validation_dataset = reformat(validation_batch, labels)
            batch_valid_data = []
            batch_valid_labels = []
            for plane, label in validation_dataset:
                batch_valid_data.append(plane)
                batch_valid_labels.append(label)
            if len(batch_valid_data) != constants.BATCH_SIZE:
                print("bad sizes validation dataset")
                print(len(batch_valid_data))
                continue
            feed_dict_valid = {tf_train_dataset: batch_valid_data, tf_train_labels: batch_valid_labels}
            summary, l, predictions_valid = sess.run([merged, loss, train_prediction], feed_dict=feed_dict_valid)
            test_writer.add_summary(summary, step)

            print('Validation accuracy: %.1f%%' % accuracy(
                predictions_valid, batch_valid_labels))
        # save progress every 500 iterations
        if step % 10 == 0 and step > 0:
            print("saving model")
            saver.save(sess, constants.CHECKPOINT_DIRECTORY + '/chess-dqn', global_step=global_step)


if __name__ == '__main__':
    main()
