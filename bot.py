from __future__ import print_function

import os
import fnmatch
import numpy as np
import tensorflow as tf

import atexit
import model
import constants
import chess
import encoder

tf_prediction, policy_logits, value = model.model(1, False)
# Predictions for the model.
policy = tf.nn.softmax(policy_logits)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state(constants.CHECKPOINT_DIRECTORY)
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("Successfully loaded:", checkpoint.model_checkpoint_path)


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
    zeros = np.full((constants.IMAGE_SIZE, constants.IMAGE_SIZE), 0, dtype=int)

    planes = np.vstack((np.copy(board_pieces), np.copy(board_white),
                        np.copy(board_black), np.copy(board_blank),
                        np.copy(current_player), np.copy(extra),
                        np.copy(move_number), np.copy(zeros)))
    planes = np.reshape(planes, (1, constants.IMAGE_SIZE, constants.IMAGE_SIZE,
                                 constants.FEATURE_PLANES))
    return planes


def board_value(board, color):
    # We get the movement prediction
    game_state = reformat(board.fen())
    feed_dict = {tf_prediction: game_state}
    result_value = sess.run([value], feed_dict=feed_dict)[0][0][0]
    #print("value = " + str(result_value))
    if color == chess.BLACK:
        result_value = -result_value
    return result_value


def board_value_dumb(board, color):
    value = 0.0
    for x in range(0, 8):
        for y in range(0, 8):
            piece = board.piece_at(chess.square(x, y))
            piece_value = 0.0
            if piece == None:
                piece_value = 0.0
            elif piece.piece_type == chess.PAWN:
                piece_value = 1.0
            elif piece.piece_type == chess.ROOK:
                piece_value = 6.0
            elif piece.piece_type == chess.KNIGHT:
                piece_value = 4.5
            elif piece.piece_type == chess.BISHOP:
                piece_value = 4.5
            elif piece.piece_type == chess.QUEEN:
                piece_value = 9.0
            elif piece.piece_type == chess.KING:
                piece_value = 9999.0

            if piece and piece.color != color:
                piece_value = -piece_value
            value += piece_value
    return value


def recurse_value(board, depth=0, max_depth=4):
    #print((" " * depth) + str(depth))

    whose_turn = board.turn
    best_move = None
    max_width = 3
    best_value = -99999.0
    count = 0
    value = 0
    if depth == max_depth:
        return None, board_value(board, whose_turn)
    for move in get_policy_moves(board):
        move = move[0]
        board.push(move)
        _, value = recurse_value(board, depth + 1, max_depth)
        value = -value
        if best_move == None or value > best_value:
            best_move = move
            best_value = value
        board.pop()
        count += 1
        if count >= max_width:
            break
    return best_move, value


def get_move(board, max_depth=5):
    move, _ = recurse_value(board, depth=0, max_depth=max_depth)
    return move


def get_policy_moves(board):
    # We get the movement prediction
    game_state = reformat(board.fen())
    feed_dict = {tf_prediction: game_state}
    predictions = sess.run([policy], feed_dict=feed_dict)
    #print(len(predictions[0][0]))
    legal_moves = []
    derp = {}
    # for i in range(0, constants.LABEL_SIZE):
    #     policy_value = predictions[0][0][i]
    #     if policy_value > 0.0:
    #         move = encoder.move_from_one_hot_index(i)
    #         print("OMG", move, policy_value)
    for move in board.legal_moves:
        policy_value = predictions[0][0][encoder.one_hot_index_from_move(move)]
        #        if policy_value != 0.0:
        #            print(move, policy_value)
        derp[move] = policy_value

    result = sorted(derp.iteritems(), key=lambda (k, v): (-v, k))
    #   print("******************************")
    # for k, v in result:
    #     print(k, v)
    return result
