"""Playing script for the network."""

from __future__ import print_function

import os
import fnmatch
import numpy as np
import tensorflow as tf

import atexit
import model
LABELS_DIRECTORY = './labels'
IMAGE_SIZE = 8
FEATURE_PLANES = 8
LABEL_SIZE = 6100
FILTERS = 128
HIDDEN = 512

labels = []


tf_prediction, logits = model.model(1)
# Predictions for the model.
train_prediction = tf.nn.softmax(logits)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("logdir")
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


def replace_tags(board):
    board_san = board.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    for i in range(len(board.split(" "))):
        if i > 0 and board.split(" ")[i] != '':
            board_san += " " + board.split(" ")[i]
    return board_san


def reformat(game):
    board_state = replace_tags(game.replace("/", ""))
    # All pieces plane
    board_pieces = list(board_state.split(" ")[0])
    board_pieces = [ord(val) for val in board_pieces]
    board_pieces = np.reshape(board_pieces, (IMAGE_SIZE, IMAGE_SIZE))
    # Only spaces plane
    board_blank = [int(val == '1') for val in board_state.split(" ")[0]]
    board_blank = np.reshape(board_blank, (IMAGE_SIZE, IMAGE_SIZE))
    # Only white plane
    board_white = [int(val.isupper()) for val in board_state.split(" ")[0]]
    board_white = np.reshape(board_white, (IMAGE_SIZE, IMAGE_SIZE))
    # Only black plane
    board_black = [int(not val.isupper() and val != '1') for val in board_state.split(" ")[0]]
    board_black = np.reshape(board_black, (IMAGE_SIZE, IMAGE_SIZE))
    # One-hot integer plane current player turn
    current_player = board_state.split(" ")[1]
    current_player = np.full((IMAGE_SIZE, IMAGE_SIZE), int(current_player == 'w'), dtype=int)
    # One-hot integer plane extra data
    extra = board_state.split(" ")[4]
    extra = np.full((IMAGE_SIZE, IMAGE_SIZE), int(extra), dtype=int)
    # One-hot integer plane move number
    move_number = board_state.split(" ")[5]
    move_number = np.full((IMAGE_SIZE, IMAGE_SIZE), int(move_number), dtype=int)
    # Zeros plane
    zeros = np.full((IMAGE_SIZE, IMAGE_SIZE), 0, dtype=int)

    planes = np.vstack((np.copy(board_pieces),
                        np.copy(board_white),
                        np.copy(board_black),
                        np.copy(board_blank),
                        np.copy(current_player),
                        np.copy(extra),
                        np.copy(move_number),
                        np.copy(zeros)))
    planes = np.reshape(planes, (1, IMAGE_SIZE, IMAGE_SIZE, FEATURE_PLANES))
    return planes

labels = read_labels(LABELS_DIRECTORY, "*.txt")

def get_move(board):
    # We get the movement prediction
    game_state = reformat(board.fen())
    feed_dict = {tf_prediction: game_state}
    predictions = sess.run([train_prediction], feed_dict=feed_dict)
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(board.san(move))
    legal_labels = [int(mov in legal_moves) for mov in labels]
    move_san = labels[np.argmax(predictions[0] * legal_labels)]
    return board.parse_san(move_san)
