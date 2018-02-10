from __future__ import print_function

import os
import fnmatch
import numpy as np
import tensorflow as tf

import atexit
import tensorflow as tf
import model
import constants
import chess
import encoder
from mcts import MCTS


class TreeBot:

    def __init__(self, sess, name):
        self.name = name
        self.sess = sess
        self.tf_prediction, self.policy_logits, self.value = model.model(
            1, False, prefix=name)
        # Predictions for the model.
        self.policy = tf.nn.softmax(self.policy_logits)

        # Initialize session all variables
        self.saver = tf.train.Saver()

        self.rename(constants.CHECKPOINT_DIRECTORY, name)
        self.mcts = MCTS(self, 200, 0)

    def replace_tags(self, board):
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

    def reformat(self, game):
        board_state = self.replace_tags(game.replace("/", ""))
        # All pieces plane
        board_pieces = list(board_state.split(" ")[0])
        board_pieces = [ord(val) for val in board_pieces]
        board_pieces = np.reshape(board_pieces,
                                  (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only spaces plane
        board_blank = [int(val == '1') for val in board_state.split(" ")[0]]
        board_blank = np.reshape(board_blank,
                                 (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only white plane
        board_white = [int(val.isupper()) for val in board_state.split(" ")[0]]
        board_white = np.reshape(board_white,
                                 (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # Only black plane
        board_black = [
            int(not val.isupper() and val != '1')
            for val in board_state.split(" ")[0]
        ]
        board_black = np.reshape(board_black,
                                 (constants.IMAGE_SIZE, constants.IMAGE_SIZE))
        # One-hot integer plane current player turn
        current_player = board_state.split(" ")[1]
        current_player = np.full(
            (constants.IMAGE_SIZE, constants.IMAGE_SIZE),
            int(current_player == 'w'),
            dtype=int)
        # One-hot integer plane extra data
        extra = board_state.split(" ")[4]
        extra = np.full(
            (constants.IMAGE_SIZE, constants.IMAGE_SIZE), int(extra), dtype=int)
        # One-hot integer plane move number
        move_number = board_state.split(" ")[5]
        move_number = np.full(
            (constants.IMAGE_SIZE, constants.IMAGE_SIZE),
            int(move_number),
            dtype=int)
        # Zeros plane
        zeros = np.full(
            (constants.IMAGE_SIZE, constants.IMAGE_SIZE), 0, dtype=int)

        planes = np.vstack((np.copy(board_pieces), np.copy(board_white),
                            np.copy(board_black), np.copy(board_blank),
                            np.copy(current_player), np.copy(extra),
                            np.copy(move_number), np.copy(zeros)))
        planes = np.reshape(planes,
                            (1, constants.IMAGE_SIZE, constants.IMAGE_SIZE,
                             constants.FEATURE_PLANES))
        return planes

    def get_move(self, board):
        move_index = np.argmax(self.mcts.getActionProb(board, temp=0))
        return encoder.move_from_one_hot_index(move_index)

    def predict(self, board):
        game_state = self.reformat(board.fen())
        feed_dict = {self.tf_prediction: game_state}
        predictions, board_value = self.sess.run(
            [self.policy, self.value], feed_dict=feed_dict)
        policy_actions = predictions[0]
        board_value = board_value[0][0]
        return policy_actions, board_value

    def rename(self, checkpoint_dir, add_prefix, dry_run=False):
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            new_name = add_prefix + "/" + var_name

            print('Renaming %s to %s.' % (var_name, new_name))
            # Rename the variable
            var = tf.Variable(var, name=new_name)

            # if not dry_run:
            #     # Save the variables
            #     saver = tf.train.Saver()
            #     sess.run(tf.global_variables_initializer())
            #     saver.save(sess, checkpoint.model_checkpoint_path)
