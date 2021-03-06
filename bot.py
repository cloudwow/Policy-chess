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


class Bot:

    def __init__(self, sess, name, max_depth, max_width):
        self.name = name
        self.sess = sess
        self.max_depth = max_depth
        self.max_width = max_width
        self.tf_prediction, self.policy_logits, self.value = model.model(
            1, False, prefix=name)
        # Predictions for the model.
        self.policy = tf.nn.softmax(self.policy_logits)

        # Initialize session all variables
        self.saver = tf.train.Saver()

        self.rename(constants.CHECKPOINT_DIRECTORY, name)

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

    # def board_value(self, board, color):
    #     # We get the movement prediction
    #     game_state = self.reformat(board.fen())
    #     feed_dict = {self.tf_prediction: game_state}
    #     result_value = self.sess.run([self.value], feed_dict=feed_dict)[0][0][0]
    #     #print("value = " + str(result_value))
    #     if color == chess.BLACK:
    #         result_value = -result_value
    #     return result_value

    def recurse_value(self, board, depth=0):

        whose_turn = board.turn
        best_move = None
        best_value = -99999.0
        count = 0
        value = 0
        moves, board_value = self.get_policy_moves(board)
        if board.turn == chess.BLACK:
            board_value = -board_value
        n = 1.0
        if depth == self.max_depth:
            return None, board_value
        for move in moves:
            move = move[0]
            board.push(move)
            _, value = self.recurse_value(board, depth=depth + 1)
            value = -value
            board_value += value
            n += 1.0
            if best_move == None or value > best_value:
                best_move = move
                best_value = value
            board.pop()
            count += 1
            if count >= self.max_width:
                break
        board_value /= n
        return best_move, best_value

    def get_move(self, board):
        move, _ = self.recurse_value(board)
        return move

    def predict(self, board):
        game_state = self.reformat(board.fen())
        feed_dict = {self.tf_prediction: game_state}
        predictions, board_value = self.sess.run(
            [self.policy, self.value], feed_dict=feed_dict)
        policy_actions = predictions[0]
        board_value = board_value[0][0]
        return policy_actions, board_value

    def get_policy_moves(self, board):
        # We get the movement prediction
        game_state = self.reformat(board.fen())
        feed_dict = {self.tf_prediction: game_state}
        predictions, board_value = self.sess.run(
            [self.policy, self.value], feed_dict=feed_dict)
        predictions = predictions[0]
        board_value = board_value[0][0]

        #print(len(predictions[0][0]))
        legal_moves = []
        derp = {}
        # for i in range(0, constants.LABEL_SIZE):
        #     policy_value = predictions[0][0][i]
        #     if policy_value > 0.0:
        #         move = encoder.move_from_one_hot_index(i)
        #         print("OMG", move, policy_value)
        for move in board.legal_moves:
            policy_value = predictions[encoder.one_hot_index_from_move(move)]
            #        if policy_value != 0.0:
            #            print(move, policy_value)
            derp[move] = policy_value

        result = sorted(derp.iteritems(), key=lambda (k, v): (-v, k))
        #   print("******************************")
        # for k, v in result:
        #     print(k, v)
        return result, board_value

    def rename(self, checkpoint_dir, add_prefix, dry_run=False):
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            new_name = add_prefix + "/" + var_name

            # Rename the variable
            var = tf.Variable(var, name=new_name)

            # if not dry_run:
            #     # Save the variables
            #     saver = tf.train.Saver()
            #     sess.run(tf.global_variables_initializer())
            #     saver.save(sess, checkpoint.model_checkpoint_path)
